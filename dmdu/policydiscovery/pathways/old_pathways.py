"""
This module is used for the pathways of optimized policies.
"""

from ema_workbench import Policy
import matplotlib.pyplot as plt
from model.pyrice import *
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import time

clr_palette = [sns.color_palette("YlGn", 15)[10], sns.cubehelix_palette(8)[6]]


def load_policies_of_one_problem_formulation(
    file="NORDHAUS_SUFFICIENTARIAN_100000_results.csv",
):
    """
    Loads and returns the optimized policies for one problem formulation.
    @param file: string: file to load
    @return:
        policies: list of Policy objects
    """

    directory = os.path.dirname(os.getcwd())
    folder = "/results_formatted/"

    df_policies = pd.read_csv(directory + folder + file).iloc[:, 1:4]

    policies = [
        Policy(f"Nordhaus_Sufficientarian_{index}", **row)
        for index, row in df_policies.iterrows()
    ]

    return policies


def get_region_pop_gr():
    """
    Get region_pop_gr.
    @return:
        region_pop_gr
    """
    RICE_POP_gr = pd.read_excel(
        "./inputdata/RICE_2010_base_000.xlsm", sheet_name="Pop_gr"
    )

    a = []
    for i in range(31):
        if i == 0:
            a.append("region")
        k = 2005 + 10 * i
        k = str(k)
        a.append(k)

    region_pop_gr = RICE_POP_gr.iloc[10:22, 3:35]
    region_pop_gr.columns = a
    region_pop_gr = region_pop_gr.set_index("region")

    return region_pop_gr


def transform_output(output):
    """

    @param output:
    @return:
    """
    region_pop_gr = get_region_pop_gr()

    output = pd.DataFrame(
        data=output, index=region_pop_gr.index, columns=region_pop_gr.columns
    )
    output = output.transpose()
    output.loc[:, "Global"] = output.sum(axis=1)
    output = output.loc[:, "Global"]

    return output


def transform_output_T(output):
    """
    @param output:
    @return:
    """
    region_pop_gr = get_region_pop_gr()
    output = pd.Series(data=output, index=region_pop_gr.columns)
    output = output.transpose()

    return output


def prepare_variables(policies):
    """
    Given some policies, this function returns the variables (that we want to visualize) as dataframes.
    @param policies: list of Policy objects
    """

    output_Y_list = []
    output_E_list = []
    output_U_list = []
    output_D_list = []
    output_T_list = []

    # Run model with policy
    for policy in policies:

        model = PyRICE(
            damage_function=DamageFunction.WEITZMAN,
            welfare_function=WelfareFunction.UTILITARIAN,
        )
        sr = policy["sr"]
        miu = policy["miu"]
        irstp = policy["irstp_consumption"]
        model(sr=sr, miu=miu, irstp=irstp)

        # Obtain relevant model results_formatted
        output_Y = model.econ_model.Y
        output_E = model.econ_model.E
        output_U = model.utility_model.per_util_ww
        output_D = model.econ_model.damages
        output_T = model.climate_model.temp_atm

        output_Y = transform_output(output_Y)
        output_Y_list.append(output_Y)

        output_E = transform_output(output_E)
        output_E_list.append(output_E)

        output_U = transform_output(output_U)
        output_U_list.append(output_U)

        output_D = transform_output(output_D)
        output_D_list.append(output_D)

        output_T = transform_output_T(output_T)
        output_T_list.append(output_T)

    return output_Y_list, output_E_list, output_U_list, output_D_list, output_T_list


def visualize_one_column(output_lists, axes, column_index, problem_formulation):
    """
    @param problem_formulation:
    @param output_lists:
    @param axes:
    @param column_index:
    """
    (
        output_Y_list,
        output_E_list,
        output_U_list,
        output_D_list,
        output_T_list,
    ) = output_lists

    cols = [
        "Weitzman\nSufficientarian",
        "Weitzman\nUtilitarian",
        "Nordhaus\nSufficientarian",
        "Nordhaus\nUtilitarian",
    ]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=22)

    color = "b"
    alpha = 1.0
    xticks = 6
    linewidth = 1.5

    for output_Y in output_Y_list:
        axes[0, column_index].set_ylabel("Economic output (trillion $)")
        axes[0, column_index].xaxis.set_major_locator(MaxNLocator(xticks))
        axes[0, column_index].plot(
            output_Y,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=problem_formulation,
        )

    for output_E in output_E_list:
        axes[1, column_index].set_ylabel("Global emissions (GTon CO2)")
        axes[1, column_index].xaxis.set_major_locator(MaxNLocator(xticks))
        axes[1, column_index].plot(
            output_E,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=problem_formulation,
        )

    for output_U in output_U_list:
        axes[2, column_index].set_ylabel("Utility (W)")
        axes[2, column_index].xaxis.set_major_locator(MaxNLocator(xticks))
        axes[2, column_index].plot(
            output_U,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=problem_formulation,
        )

    for output_D in output_D_list:
        axes[3, column_index].set_ylabel("Economic damages (trillion $)")
        axes[3, column_index].xaxis.set_major_locator(MaxNLocator(xticks))
        axes[3, column_index].plot(
            output_D,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=problem_formulation,
        )

    for output_T in output_T_list:
        axes[4, column_index].set_ylabel(
            "Increase in atmospheric \ntemperature (Celsius)"
        )
        axes[4, column_index].xaxis.set_major_locator(MaxNLocator(xticks))
        axes[4, column_index].plot(
            output_T,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label=problem_formulation,
        )
        axes[4, column_index].set_xlabel("Time (years)")


def visualize(
    output_lists_WS, output_lists_WU, output_lists_NS, output_lists_NU, saving=False
):
    """
    @param saving: Boolean: True if you want to save the figure as an image
    @param output_lists_WS: list of outputs
    @param output_lists_WU: list of outputs
    @param output_lists_NS: list of outputs
    @param output_lists_NU: list of outputs
    """
    sns.set(font_scale=1.35)
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        nrows=5,
        ncols=4,
        sharex="all",
        figsize=(22, 18),
        tight_layout=True,
        sharey="row",
    )
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8
    )

    visualize_one_column(
        output_lists_WS, axes, column_index=0, problem_formulation="WS"
    )
    visualize_one_column(
        output_lists_WU, axes, column_index=1, problem_formulation="WU"
    )
    visualize_one_column(
        output_lists_NS, axes, column_index=2, problem_formulation="NS"
    )
    visualize_one_column(
        output_lists_NU, axes, column_index=3, problem_formulation="NU"
    )

    plt.show()

    if saving:
        directory = os.getcwd()
        root_directory = os.path.dirname(directory)
        visualization_folder = root_directory + "/dmdu/outputimages/"
        fig.savefig(visualization_folder + "pathways.png", dpi=200, pad_inches=0.2)


if __name__ == "__main__":

    start_time = time.time()

    policies_WS = load_policies_of_one_problem_formulation(
        "WEITZMAN_SUFFICIENTARIAN_200000_results.csv"
    )
    policies_WU = load_policies_of_one_problem_formulation(
        "WEITZMAN_UTILITARIAN_200000_results.csv"
    )
    policies_NS = load_policies_of_one_problem_formulation(
        "NORDHAUS_SUFFICIENTARIAN_200000_results.csv"
    )
    policies_NU = load_policies_of_one_problem_formulation(
        "NORDHAUS_UTILITARIAN_200000_results.csv"
    )

    output_lists_WS = prepare_variables(policies_WS)
    output_lists_WU = prepare_variables(policies_WU)
    output_lists_NS = prepare_variables(policies_NS)
    output_lists_NU = prepare_variables(policies_NU)

    visualize(
        output_lists_WS, output_lists_WU, output_lists_NS, output_lists_NU, saving=False
    )

    run_time = round(time.time() - start_time, 2)
    print(f"Run time: {run_time} seconds")
