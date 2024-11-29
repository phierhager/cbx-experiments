import cbx.utils
import cbx.utils.objective_handling
from cbx_experiments.config import ExperimentConfig, ConfigContainerDynamic
from cbx_experiments.experiment_result import ExperimentResult, ResultDynamicRun
import cbx
from cbx.scheduler import multiply
import time
import os
import numpy as np
from tqdm import tqdm
from typing import Callable, Any


def timing(method: Callable):
    """Time the execution of a method.

    Args:
        method (Callable): The method to time.
    """

    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))
        elapsed_time = endTime - startTime
        return elapsed_time, result

    return wrapper


@timing
def optimize_wrapper(
    dynamic: cbx.dynamics.ParticleDynamic, *args, **kwargs
) -> np.ndarray | Any:
    """Optimize a dynamic.

    Args:
        dynamic (cbx.dynamics.ParticleDynamic): The dynamic to optimize.

    Returns:
        np.ndarray | Any: The result of the optimization.
    """
    return dynamic.optimize(*args, **kwargs)


class Runner:
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        result_dir: str = "results",
    ) -> None:
        """Initialize the Runner.

        Args:
            experiment_config (ExperimentConfig): The experiment configuration.
            result_dir (str, optional): The directory to save the results. Defaults to "results".
        """
        self.experiment_config: ExperimentConfig = experiment_config
        self.experiment_result: ExperimentResult = ExperimentResult(
            experiment_config.experiment_name,
            experiment_config.config_opt,
            results_dynamic=[],
        )
        self.result_dir = result_dir
        self.starting_positions = (
            None  # TODO: THere was a bug here, so it is not initialized now
        )
        self.local_minimum_functions = {}  # will be filled for every funnction calculate nearest local minimum with scipy

    def run_experiment(self) -> ExperimentResult:
        """Run the experiment.

        Returns:
            ExperimentResult: The experiment result.
        """
        results_dynamic = self.run_dynamic_configs()
        self.experiment_result.results_dynamic = results_dynamic
        if self.result_dir is not None:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.experiment_result.to_csv(
                self.result_dir
                + "/"
                + "_".join(str(self.experiment_result.experiment_name).lower().split())
                + ".csv"
            )
        return self.experiment_result

    def run_dynamic_configs(self) -> list[ResultDynamicRun]:
        """Run the dynamic configurations.

        Returns:
            list[ResultDynamicRun]: The results of the dynamic configurations.
        """
        results_dynamic = []
        opt_dict = self.get_opt_dict()
        for config_container_dynamic in tqdm(
            self.experiment_config.config_container_dynamic_gen
        ):
            function_name = config_container_dynamic.name_f
            if function_name not in self.local_minimum_functions.keys():
                if function_name == "ackley":
                    self.local_minimum_functions[function_name] = 2.181
                elif function_name == "rastrigin":
                    self.local_minimum_functions[function_name] = 1
                else:
                    print(
                        f"WARNING: The nearest local minimum for {config_container_dynamic.f} and dimension is not yet implemented. The success will be calculated by if the best value is less than 1. If you want to implement it, please add it to the Runner class in the run_dynamic_configs method."
                    )
            result = self.run_dynamic_config(config_container_dynamic, opt_dict)
            results_dynamic.append(result)
        return results_dynamic

    def run_dynamic_config(
        self,
        config_container_dynamic: ConfigContainerDynamic,
        opt_dict: dict,
    ) -> ResultDynamicRun:
        """Run a dynamic configuration.

        Args:
            config_container_dynamic (ConfigContainerDynamic): The dynamic configuration.
            opt_dict (dict): The optimization dictionary.

        Returns:
            ResultDynamicRun: The result of the dynamic configuration.
        """
        f = config_container_dynamic.f
        config_dynamic = config_container_dynamic.config_dynamic
        # this also pops the x from the config_dynamic in the config_container_dynamic
        config_dynamic.pop("x")
        dynamic = config_container_dynamic.dynamic(
            f, x=self.starting_positions, **config_dynamic
        )
        time, best_x = optimize_wrapper(dynamic, **opt_dict)  # optimize
        best_f = f(best_x)
        success = np.where(
            best_f < self.local_minimum_functions[config_container_dynamic.name_f], 1, 0
        )
        return ResultDynamicRun(
            name_dynamic=config_container_dynamic.name_dynamic,
            name_f=config_container_dynamic.name_f,
            index_config=config_container_dynamic.index_config,
            config_dynamic=config_dynamic,
            time=time,
            best_f=best_f,
            best_x=best_x,
            success=success,
        )

    def get_opt_dict(self) -> dict:
        """Get the optimization dictionary.

        Returns:
            dict: The optimization dictionary.
        """
        _config_opt = self.experiment_config.config_opt
        return {
            "sched": multiply(
                factor=_config_opt["sched"]["factor"],
                maximum=_config_opt["sched"]["maximum"],
            ),
            "print_int": _config_opt["print_int"],
        }
