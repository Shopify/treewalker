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

variable "bench_suite" {
  description = "Benchmark suite: 'paper' (factorial grid, --grid all) or 'rebuttal' (scenario-analysis + chunked-G runs only)"
  type        = string
  default     = "paper"
}

variable "expedia_parquet" {
  description = "Path to expedia.parquet built by treewalker-exp fetch-expedia; required for bench_suite = \"paper\" (default: ../experiments/data/expedia.parquet)"
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
      image        = "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      role         = "trainer"
    }
    arm = {
      machine_type = "c4a-highmem-16"
      image        = "ubuntu-os-cloud/ubuntu-2404-lts-arm64"
      role         = "benchmarker"
    }
  }
}
