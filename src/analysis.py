import logging
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gboml import GbomlGraph

class GbomlModel:

    def run_model(self, 
        model_txt: str, timehorizon: int = 8760, num_households: int = 10, num_years=1
    ):
        gboml_model = GbomlGraph(timehorizon)
        gboml_model.add_global_parameter("num_households", num_households)
        gboml_model.add_global_parameter("number_years_horizon", num_years)
        nodes, edges, _ = gboml_model.import_all_nodes_and_edges(f"{model_txt}", cache=False)
        gboml_model.add_nodes_in_model(*nodes)
        gboml_model.add_hyperedges_in_model(*edges)
        gboml_model.build_model()
        solution = gboml_model.solve_gurobi(details=True)
        (solution_flat, objective, status, solver_info, constr_info, var_info) = solution
        solution_dict = gboml_model.turn_solution_to_dictionary(
            solution=solution_flat,
            solver_data=solver_info,
            status=status,
            objective=objective,
            constraint_info=constr_info,
            variables_info=var_info,
        )
        return solution, solution_dict


    def show_results(self, solution_dict: dict, days: int = 7):
        """
        This function reads the solution that gboml creates as an
        output and prints the results in the stdout for better readability
        and faster interpretation of the results. For scalar results like
        investment costs, it just prints the result. For continuous variables
        it creates and prints a plot over the first seven days but the number
        of days for the visualization can be given as a parameter and thus a
        visualization for a longer period of time can be created.

        Parameters
        ----------
        solution_dict : dict
            The solution dictionary that gboml creates as an output.
        days : int
            The number of days the visualizations should be created for.
        """
        print(f"Objective result: {str(solution_dict['solution']['objective'])}")
        print(f"Objective status: {str(solution_dict['solution']['status'])}")
        elements = solution_dict["solution"]["elements"]
        for element in elements.keys():
            for variable in elements[element]["variables"].keys():
                var_values = elements[element]["variables"][variable]["values"]
                if len(var_values) == 1:
                    result = f"Node {element}: {variable}: {var_values[0]}"
                    print(result)
                else:
                    print(f"Node {element}: {variable}: (first week)")
                    plt.plot(var_values[: 24 * days])
                    plt.show()


    def get_results_new(self, scenarios: dict) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        This function runs gboml with all of the given scenarios and writes
        the results to two separate dataframes.

        Parameters
        ----------
        scenarios : dict
            A dict containing all the models you want to run and the respective
            number of households, e.g. {'model.txt': 10, 'benchmark.txt': 10}

        Returns
        -------
        Tuple
            objective_result: pd.DataFrame contains the objective investment costs overall and per node
            result: pd.DataFrame contains the optimal parameters for each node
            solution_dict: raw dictionary of the solution, contains schedules with hourly resolution
        """
        result = pd.DataFrame(
            columns=["execution_time", "num_households", "self_consumption(%)"],
            index=scenarios.keys(),
        )
        objective_result = pd.DataFrame(
            columns=[
                "execution_time",
                "objective",
                "obj_per_household",
                "status",
                "num_households",
            ],
            index=scenarios.keys(),
        )
        for model in scenarios.keys():
            logging.info(f"Runnig {model}.")
            start_time = time.time()
            num_households = scenarios[model]
            _, solution_dict = self.run_model(model, num_households=num_households)
            end_time = time.time() - start_time
            (
                objective_result.loc[model, "execution_time"],
                result.loc[model, "execution_time"],
            ) = (end_time, end_time)
            objective_result.loc[model, "objective"] = solution_dict["solution"][
                "objective"
            ]
            if solution_dict["solution"]["objective"] is not None:
                objective_result.loc[model, "obj_per_household"] = (
                    solution_dict["solution"]["objective"] / num_households
                )
                objective_result.loc[model, "status"] = solution_dict["solution"]["status"]
                result.loc[model, "num_households"] = num_households
                result.loc[model, "self_consumption(%)"] = self._calculate_self_consumption(
                    solution_dict
                )
                objective_result.loc[model, "num_households"] = num_households
                elements = solution_dict["solution"]["elements"]
                for element in elements.keys():
                    if "objectives" in elements[element].keys():
                        objective_result.loc[model, f"{element}: objective"] = elements[
                            element
                        ]["objectives"]["unnamed"][0]
                        objective_result.loc[model, f"{element}: obj_per_household"] = (
                            elements[element]["objectives"]["unnamed"][0] / num_households
                        )
                    for variable in elements[element]["variables"].keys():
                        var_values = elements[element]["variables"][variable]["values"]
                        if len(var_values) == 1:
                            column = f"{element}: {variable}"
                            result.loc[model, column] = var_values[0]
            else:
                continue
        return objective_result, result, solution_dict


    def _calculate_self_consumption(self, solution_dict):
        """ """
        el = solution_dict["solution"]["elements"]
        import_el = np.sum(
            el["DISTRIBUTION_EL"]["variables"]["electricity_import"]["values"]
        )
        demand_el = 0
        demand_gas, import_heat, demand_heat, import_gas = 0, 0, 0, 0
        if "DISTRIBUTION_GAS" in solution_dict["solution"]["elements"].keys():
            import_gas = np.sum(
                el["DISTRIBUTION_GAS"]["variables"]["gas_import_amount"]["values"]
            )
        if "CHP_PLANT" in solution_dict["solution"]["elements"].keys():
            demand_gas = np.sum(el["CHP_PLANT"]["variables"]["consumption_gas"]["values"])
        demand = demand_el + demand_gas
        import_all = import_el + import_gas
        return round((1 - import_all / demand) * 100, 2)


    def create_config(self, num_households: int):
        directory = "../models/scenarios/"
        print(directory)
        # Get all files in the directory
        model_list = os.listdir(directory)
        # Filter out directories from the file list and prepend directory path
        model_list_with_path = [
            os.path.join(directory, file)
            for file in model_list
            if os.path.isfile(os.path.join(directory, file))
        ]
        config = {key: num_households for key in model_list_with_path}
        return config
