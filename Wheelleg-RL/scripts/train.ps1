param(
  [ValidateSet('flat','rough','recovery')][string]$Stage='flat',
  [int]$NumEnvs=2048,
  [int]$MaxIterations=5,
  [switch]$NoWandb
)
$tasks = @{ flat='Wheelleg-Flat-v0'; rough='Wheelleg-Rough-v0'; recovery='Wheelleg-Recovery-v0' }
$task = $tasks[$Stage]
$args = @('run','train',$task,'--env.scene.num-envs',$NumEnvs,'--agent.max_iterations',$MaxIterations)
if ($NoWandb) { $env:WANDB_MODE='disabled' }
uv @args
