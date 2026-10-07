variable "project" {
  description = "GCP project ID"
  type        = string
}

variable "zone" {
  description = "GCE zone for benchmark instances"
  type        = string
  default     = "us-central1-a"
}

variable "git_ref" {
  # No default: the startup script needs experiments/ and treewalker-exp, which
  # the neurips2026 tag predates. Pass a ref that contains them.
  description = "Git branch, tag, or SHA to benchmark; it must contain experiments/ and treewalker-exp"
  type        = string
}

variable "suites" {
  description = "Suites from experiments/grids.toml to prepare and run, in order; run acceptance alone first"
  type        = list(string)
  default     = ["factorial", "ablation"]
}

variable "layout_check" {
  description = "Also time TreeWalker on the acceptance cells in a build with 64-byte function alignment (layout sensitivity)"
  type        = bool
  default     = false
}

variable "apt_snapshot" {
  description = "Ubuntu archive snapshot (YYYYMMDDTHHMMSSZ) every apt operation on the VMs uses, so the compiler and tools match across deployments and the cached baselines stay valid; empty for the live archive. The default is the archive the 2026-10-05 shakedown factorial installed from."
  type        = string
  default     = "20261005T111000Z"
}

variable "gate_ref" {
  description = "A candidate ref for the kernel gate: instead of the suites' runs, time TreeWalker built from git_ref (A) and from this ref (B) in alternating processes, with default and 64-byte function alignment; empty for none"
  type        = string
  default     = ""
}

variable "cache_bucket" {
  description = "An existing bucket that keeps prepared models and compiled baselines across deployments, so a rerun does not retrain or recompile; empty for none. Create it outside Terraform, so destroy leaves it."
  type        = string
  default     = ""
}

variable "expedia_parquet" {
  description = "Path to expedia.parquet built by treewalker-exp fetch-expedia; needed when a suite has Expedia cells (default: ../experiments/data/expedia.parquet)"
  type        = string
  default     = ""
}

variable "network" {
  description = "VPC network name"
  type        = string
  default     = "default"
}

variable "subnet" {
  description = "Subnetwork for instances (must belong to var.network)"
  type        = string
  default     = "default"
}

variable "machines" {
  description = "Map of machine configs to benchmark"
  type = map(object({
    machine_type = string
    image        = string
    role         = string # "trainer" (trains + uploads artifacts) or "benchmarker" (downloads + benchmarks)
  }))
  default = {
    intel = {
      machine_type = "c4-standard-32"
      image        = "ubuntu-os-cloud/ubuntu-2604-resolute-amd64-v20260918"
      role         = "trainer"
    }
    arm = {
      machine_type = "c4a-highmem-16"
      image        = "ubuntu-os-cloud/ubuntu-2604-resolute-arm64-v20260918"
      role         = "benchmarker"
    }
  }
}
