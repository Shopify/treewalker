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

# The Expedia data cannot be redistributed; the factorial grid needs the
# parquet built locally by fetch_expedia.py, uploaded for the trainer.
resource "google_storage_bucket_object" "expedia" {
  count  = var.bench_suite == "paper" ? 1 : 0
  name   = "inputs/expedia.parquet"
  bucket = google_storage_bucket.bench.name
  source = local.expedia_parquet

  lifecycle {
    precondition {
      condition     = fileexists(local.expedia_parquet)
      error_message = "bench_suite = \"paper\" needs expedia.parquet; build it with experiments/scripts/fetch_expedia.py."
    }
  }
}

resource "google_storage_bucket_object" "source" {
  name   = "source-${random_id.suffix.hex}.tar.gz"
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
    threads_per_core = 1 # disable SMT for stable single-threaded benchmarks
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
      git_ref           = var.git_ref
      role              = each.value.role
      bench_suite       = var.bench_suite
    }
  )

  labels = {
    purpose = "benchmark"
    arch    = each.key
  }
}
