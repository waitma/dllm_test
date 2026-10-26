"""Persist validation-driven LR reductions across steps and checkpoint restores.

Activate pllm; USE_TF=0 python -B -m pytest
/project_c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_diffusion_training_guard.py -q.
This controller changes optimization only; it does not change the noise objective.
"""
import copy
import math


class ValidationLRCeiling:
    """Cap an existing schedule after consecutive non-improving validations.

    The cap never increases. Per-step cosine/warmup cannot undo a reduction.
    Validation is counted once per optimizer step, including after resume.
    """

    def __init__(self, scheduler, config, *, best=None, initial_scale=1.):
        required = {"factor", "patience", "min_lr", "relative_threshold"}
        if set(config) != required:
            raise ValueError("Unexpected validation LR guard fields")
        factor, patience = config["factor"], config["patience"]
        floor, threshold = config["min_lr"], config["relative_threshold"]
        if (not 0 < factor < 1 or type(patience) is not int or patience < 1
                or not math.isfinite(floor) or floor <= 0
                or not 0 <= threshold < 1 or not 0 < initial_scale <= 1):
            raise ValueError("Invalid validation LR guard configuration")
        if best is not None and (not math.isfinite(best) or best < 0):
            raise ValueError("Invalid reference validation loss")
        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer
        self.config = copy.deepcopy(config)
        self.best = best
        self.bad_validations = 0
        self.last_validation_step = -1
        self.reductions = 0
        # A None cap preserves the original warmup until a reduction is needed.
        self.ceilings = [None if initial_scale == 1 else min(
            group["lr"], max(floor, group["lr"] * initial_scale))
            for group in self.optimizer.param_groups]
        self._apply()

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch

    def _apply(self):
        for group, base_lr, ceiling in zip(
                self.optimizer.param_groups, self.scheduler.get_last_lr(), self.ceilings):
            group["lr"] = base_lr if ceiling is None else min(base_lr, ceiling)

    def step(self):
        self.scheduler.step()
        self._apply()

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def observe(self, loss, step):
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("Validation loss must be finite and nonnegative")
        if step != self.last_epoch or step < self.last_validation_step:
            raise ValueError("Validation/scheduler step mismatch")
        if step == self.last_validation_step:
            return False
        self.last_validation_step = step
        if self.best is None or loss < self.best * (1 - self.config["relative_threshold"]):
            self.best = loss
            self.bad_validations = 0
            return False
        self.bad_validations += 1
        if self.bad_validations < self.config["patience"]:
            return False
        changed = False
        for index, group in enumerate(self.optimizer.param_groups):
            current = group["lr"]
            ceiling = min(current, max(self.config["min_lr"], current * self.config["factor"]))
            changed |= ceiling < current
            self.ceilings[index] = ceiling
        self.bad_validations = 0
        self.reductions += int(changed)
        self._apply()
        return changed

    def state_dict(self):
        return copy.deepcopy(dict(schema="diffusion.validation_lr_ceiling.v1",
            scheduler=self.scheduler.state_dict(), config=self.config, best=self.best,
            bad_validations=self.bad_validations, last_validation_step=self.last_validation_step,
            reductions=self.reductions, ceilings=self.ceilings))

    def load_state_dict(self, state):
        if (state.get("schema") != "diffusion.validation_lr_ceiling.v1"
                or state["config"] != self.config
                or len(state["ceilings"]) != len(self.optimizer.param_groups)):
            raise ValueError("Validation LR guard checkpoint differs")
        self.scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
        for name in ("best", "bad_validations", "last_validation_step", "reductions", "ceilings"):
            setattr(self, name, copy.deepcopy(state[name]))
        self._apply()
