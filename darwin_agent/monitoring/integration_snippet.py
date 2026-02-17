"""Darwin dashboard integration snippet."""

from darwin_agent.monitoring.execution_audit import ExecutionAudit
from dashboard.app import set_execution_audit, set_bot_runner
from dashboard.bot_runtime import DarwinRuntime


audit = ExecutionAudit(log_dir="logs/audit")
set_execution_audit(audit)


def runner(stop_event, controller):
    runtime = DarwinRuntime(controller=controller, audit=audit)
    runtime.start()


set_bot_runner(runner)
