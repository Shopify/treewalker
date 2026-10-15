# The camera-ready campaign, after the acceptance suite has run alone: the
# paper's cells, the ablation suite twice as two runs, with the factorial between
# them so their spread spans the run, E1 (scenario-v1) and the layout check,
# every TreeWalker forest with exact sums off, then the re-timing of every cell
# whose interval is wider than ±2% (retime = true), on the same VMs.
#   terraform apply -var project=YOUR_PROJECT -var git_ref=$(git rev-parse HEAD) -var-file=camera-ready.tfvars
zone         = "us-east4-a"
suites       = ["acceptance", "camera-ready-ablation@1", "camera-ready", "camera-ready-ablation@2", "scenario-v1"]
layout_check = true
retime       = true
run_settings = ["exact_sums=false"]
