output "instances" {
  description = "Summary of benchmark instances"
  value = {
    for k, inst in google_compute_instance.bench : k => {
      name         = inst.name
      machine_type = inst.machine_type
      zone         = inst.zone
    }
  }
}

output "ssh_commands" {
  description = "SSH into each benchmark instance"
  value = {
    for k, inst in google_compute_instance.bench : k =>
    "gcloud compute ssh ${inst.name} --project=${var.project} --zone=${inst.zone}"
  }
}

output "tail_logs" {
  description = "Tail benchmark log on each instance"
  value = {
    for k, inst in google_compute_instance.bench : k =>
    "gcloud compute ssh ${inst.name} --project=${var.project} --zone=${inst.zone} --command 'tail -f /var/log/treewalker-bench.log'"
  }
}

output "check_done" {
  description = "Check if benchmarks have finished"
  value = {
    for k, inst in google_compute_instance.bench : k =>
    "gcloud compute ssh ${inst.name} --project=${var.project} --zone=${inst.zone} --command 'cat /home/bench/results/DONE 2>/dev/null || cat /home/bench/results/ERROR 2>/dev/null || echo RUNNING'"
  }
}

output "scp_results" {
  description = "Download results from each instance"
  value = {
    for k, inst in google_compute_instance.bench : k =>
    "gcloud compute scp --recurse ${inst.name}:/home/bench/results/ ./results/bench_results_${k}/ --project=${var.project} --zone=${inst.zone}"
  }
}
