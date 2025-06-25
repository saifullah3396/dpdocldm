from ignite.engine import EventEnum


class OptimizerEvents(EventEnum):
    optimizer_step = "optimizer_step"
