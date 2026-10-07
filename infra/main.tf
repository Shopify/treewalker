resource "random_id" "suffix" {
  byte_length = 3
}

# --- Source upload to GCS ---

resource "google_storage_bucket" "bench" {
  name                        = "treewalker-bench-${random_id.suffix.hex}"
  location                    = regex("^(.*)-[a-z]$", var.zone)[0]
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition { age = 7 }
    action { type = "Delete" }
  }
}

resource "terraform_data" "source_archive" {
  triggers_replace = [timestamp()]

  provisioner "local-exec" {
    command = "git -C .. archive --format=tar.gz --output=${abspath(path.module)}/.terraform/source.tar.gz ${var.git_ref}"
  }
}

locals {
  expedia_parquet = var.expedia_parquet != "" ? var.expedia_parquet : "${path.module}/../experiments/data/expedia.parquet"
}

locals {
  # Counters at a fixed frequency: the all-core maximum instead of opportunistic
  # turbo, and the PMU level that exposes core events (sweep_bench preflight
  # checks the event group).
  turbo_mode = "ALL_CORE_MAX"
  pmu_level  = "STANDARD"
  # GCE offers ALL_CORE_MAX on C4 and C4N only. C4A rejects it: Arm processors
  # always run at their all-core turbo frequency.
  turbo_series = ["c4", "c4n"]
  effective_turbo = {
    for k, m in var.machines : k => (
      contains(local.turbo_series, split("-", m.machine_type)[0])
      ? local.turbo_mode
      : "unset: Arm runs at its all-core turbo frequency"
    )
  }
  # Suites with Expedia cells need the parquet; acceptance and smoke do not.
  needs_expedia = length(setsubtract(var.suites, ["acceptance", "smoke", "scenario-v1"])) > 0
}

# The Expedia data cannot be redistributed; suites with Expedia cells need the
# parquet built locally by treewalker-exp fetch-expedia, uploaded for the trainer.
resource "google_storage_bucket_object" "expedia" {
  count  = local.needs_expedia ? 1 : 0
  name   = "inputs/expedia.parquet"
  bucket = google_storage_bucket.bench.name
  source = local.expedia_parquet

  lifecycle {
    precondition {
      condition     = fileexists(local.expedia_parquet)
      error_message = "These suites need expedia.parquet; build it with uv run treewalker-exp fetch-expedia."
    }
  }
}

# The archive is written during apply, after the plan has read the old file, so the
# object's content cannot tell a new ref apart. Its name carries the ref instead: a
# new ref creates a new object, uploaded from the fresh archive, and replaces the
# instances that read it.
resource "google_storage_bucket_object" "source" {
  name   = "source-${random_id.suffix.hex}-${var.git_ref}.tar.gz"
  bucket = google_storage_bucket.bench.name
  source = "${path.module}/.terraform/source.tar.gz"

  depends_on = [terraform_data.source_archive]
}

# --- Benchmark instances ---

resource "google_compute_instance" "bench" {
  for_each     = var.machines
  name         = "treewalker-bench-${each.key}-${random_id.suffix.hex}"
  machine_type = each.value.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = each.value.image
      size  = 200
      type  = "hyperdisk-balanced"
    }
  }

  network_interface {
    network    = var.network
    subnetwork = var.subnet
    access_config {} # ephemeral public IP
  }

  advanced_machine_features {
    threads_per_core            = 1               # disable SMT for stable single-threaded benchmarks
    performance_monitoring_unit = local.pmu_level # core events, L2 included; L3 needs ENHANCED
    turbo_mode                  = startswith(local.effective_turbo[each.key], "unset") ? null : local.effective_turbo[each.key]
  }

  scheduling {
    on_host_maintenance = "TERMINATE" # no live migration during benchmarks
    automatic_restart   = false
    preemptible         = false
  }

  service_account {
    scopes = ["storage-full"] # trainer uploads artifacts, both upload results
  }

  metadata = {
    enable-oslogin = "TRUE"
  }

  metadata_startup_script = templatefile(
    "${path.module}/scripts/startup.sh",
    {
      gcs_source_uri    = "gs://${google_storage_bucket.bench.name}/${google_storage_bucket_object.source.name}"
      gcs_results_base  = "gs://${google_storage_bucket.bench.name}/results/${each.key}"
      gcs_artifacts_uri = "gs://${google_storage_bucket.bench.name}/artifacts"
      gcs_expedia_uri   = length(google_storage_bucket_object.expedia) > 0 ? "gs://${google_storage_bucket.bench.name}/${google_storage_bucket_object.expedia[0].name}" : ""
      gcs_cache_uri     = var.cache_bucket == "" ? "" : "gs://${var.cache_bucket}"
      git_ref           = var.git_ref
      role              = each.value.role
      suites            = join(" ", var.suites)
      layout_check      = var.layout_check
      apt_snapshot      = var.apt_snapshot
      machine_type      = each.value.machine_type
      image             = each.value.image
      turbo_mode        = local.effective_turbo[each.key]
      pmu_level         = local.pmu_level
    }
  )

  labels = {
    purpose = "benchmark"
    arch    = each.key
  }
}
