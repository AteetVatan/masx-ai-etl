class DAGContext:
    """
    XCom wrapper to push/pull values across tasks.
    """

    def __init__(self, kwargs):
        self.kwargs = kwargs

    def push(self, key, value):
        self.kwargs["ti"].xcom_push(key=key, value=value)

    def pull(self, key, task_id=None):
        return self.kwargs["ti"].xcom_pull(key=key, task_ids=task_id)
