import openmdao.api as om


class PlantFinance(om.ExplicitComponent):
    """
    Compute LCOE for the wind plant

    Parameters
    ----------
    machine_rating : float
        Rating of the turbine
    tcc_per_kW : float
        Turbine capital cost
    offset_tcc_per_kW : float
        Offset to turbine capital cost
    turbine_number : int
        Number of turbines at plant
    bos_per_kW : float
        Balance of system costs of the turbine
    opex_per_kW : float
        Average annual operational expenditures of the turbine
    plant_aep_in : float
        Annual Energy Production of the wind plant
    turbine_aep : float
        Annual Energy Production of the wind turbine
    wake_loss_factor : float
        The losses in AEP due to waked conditions
    fixed_charge_rate : float
        Fixed charge rate for coe calculation

    Returns
    -------
    lcoe : float
        Levelized cost of energy for the wind plant
    plant_aep : float
        Annual Energy Production of the wind plant

    """

    def initialize(self):
        self.options.declare("verbosity", default=False)

    def setup(self):

        # Inputs
        self.add_input("machine_rating", val=0.0, units="kW")
        self.add_input("tcc_per_kW", val=0.0, units="USD/kW")
        self.add_input("offset_tcc_per_kW", val=0.0, units="USD/kW")
        self.add_discrete_input("turbine_number", val=0)
        self.add_input("bos_per_kW", val=0.0, units="USD/kW")
        self.add_input("opex_per_kW", val=0.0, units="USD/kW/yr")
        self.add_input("plant_aep_in", val=0.0, units="kW*h")
        self.add_input("turbine_aep", val=0.0, units="kW*h")
        self.add_input("wake_loss_factor", val=0.15)
        self.add_input("fixed_charge_rate", val=0.075)

        self.add_output("lcoe", val=0.0, units="USD/kW/h")
        self.add_output("plant_aep", val=0.0, units="USD/kW/h")

        self.declare_partials("*", "*")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack parameters
        t_rating = inputs["machine_rating"]
        n_turbine = discrete_inputs["turbine_number"]
        tcc_per_kW = inputs["tcc_per_kW"] + inputs["offset_tcc_per_kW"]
        bos_per_kW = inputs["bos_per_kW"]
        opex_per_kW = inputs["opex_per_kW"]
        fcr = inputs["fixed_charge_rate"]
        wlf = inputs["wake_loss_factor"]
        turb_aep = inputs["turbine_aep"]
        c_turbine = tcc_per_kW * t_rating
        c_bos_turbine = bos_per_kW * t_rating
        c_opex_turbine = opex_per_kW * t_rating

        # Run a few checks on the inputs
        if n_turbine == 0:
            raise ValueError(
                "ERROR: The number of the turbines in the plant is not initialized correctly and it is currently equal to 0. Check the connections to Plant_FinanceSE"
            )

        if c_turbine == 0:
            raise ValueError(
                "ERROR: The cost of the turbines in the plant is not initialized correctly and it is currently equal to 0 USD. Check the connections to Plant_FinanceSE"
            )

        if c_bos_turbine == 0:
            print(
                "WARNING: The BoS costs of the turbine are not initialized correctly and they are currently equal to 0 USD. Check the connections to Plant_FinanceSE"
            )

        if c_opex_turbine == 0:
            print(
                "WARNING: The Opex costs of the turbine are not initialized correctly and they are currently equal to 0 USD. Check the connections to Plant_FinanceSE"
            )

        if inputs["plant_aep_in"] == 0:
            if turb_aep == 0:
                print(
                    "WARNING: AEP is not computed properly or it is not connected to PlantFinanceSE. Both turbine_aep and park_aep are currently equal to 0 Wh. Check the connections to Plant_FinanceSE"
                )
            park_aep = n_turbine * turb_aep * (1.0 - wlf)
            dpark_dtaep = n_turbine * (1.0 - wlf)
            dpark_dnturb = turb_aep * (1.0 - wlf)
            dpark_dwlf = -n_turbine * turb_aep
            dpark_dpaep = 0.0
        else:
            park_aep = inputs["plant_aep_in"]
            dpark_dpaep = 1.0
            dpark_dtaep = dpark_dnturb = dpark_dwlf = 0.0

        npr = n_turbine * t_rating  # net park rating, used in net energy capture calculation below
        dnpr_dnturb = t_rating
        dnpr_dtrating = n_turbine

        nec = park_aep / npr  # net energy rating, per COE report
        dnec_dwlf = dpark_dwlf / npr
        dnec_dtaep = dpark_dtaep / npr
        dnec_dpaep = dpark_dpaep / npr
        dnec_dnturb = dpark_dnturb / npr - dnpr_dnturb * nec / npr
        dnec_dtrating = -dnpr_dtrating * nec / npr

        icc = (c_turbine + c_bos_turbine) / t_rating  # $/kW, changed per COE report
        c_opex = (c_opex_turbine) / t_rating  # $/kW, changed per COE report

        dicc_dtrating = -icc / t_rating
        dcopex_dtrating = -c_opex / t_rating
        dicc_dcturb = dicc_dcbos = dcopex_dcopex = 1.0 / t_rating

        # compute COE and LCOE values
        lcoe = (icc * fcr + c_opex) / nec  # changed per COE report
        outputs["lcoe"] = lcoe
        outputs["plant_aep"] = park_aep

        self.J = {}
        self.J["lcoe", "tcc_per_kW"] = dicc_dcturb * fcr / nec
        self.J["lcoe", "turbine_number"] = -dnec_dnturb * lcoe / nec
        self.J["lcoe", "bos_per_kW"] = dicc_dcbos * fcr / nec
        self.J["lcoe", "opex_per_kW"] = dcopex_dcopex / nec
        self.J["lcoe", "fixed_charge_rate"] = icc / nec
        self.J["lcoe", "wake_loss_factor"] = -dnec_dwlf * lcoe / nec
        self.J["lcoe", "turbine_aep"] = -dnec_dtaep * lcoe / nec
        self.J["lcoe", "plant_aep_in"] = -dnec_dpaep * lcoe / nec
        self.J["lcoe", "machine_rating"] = (dicc_dtrating * fcr + dcopex_dtrating) / nec - dnec_dtrating * lcoe / nec

        if self.options["verbosity"]:
            print("################################################")
            print("Computation of LCoE from Plant_FinanceSE")
            print("Number of turbines in the park    %u" % n_turbine)
            print("Turbine rating                    %.2f kW" % t_rating)
            print("Turbine capital cost per kW       %.2f USD/kW" % tcc_per_kW)
            print("BoS costs per kW                  %.2f USD/kW" % bos_per_kW)
            print("Opex costs per kW                 %.2f USD/kW" % opex_per_kW)
            print("Fixed charge rate                 %.2f %%" % (fcr * 100.0))
            print("Wake loss factor                  %.2f %%" % (wlf * 100.0))
            print("AEP of the single turbine         %.2f GWh" % (turb_aep * 1.0e-006))
            print("AEP of the wind plant             %.2f GWh" % (park_aep * 1.0e-006))
            print("Initial capital costs per kW      %.2f $/kW" % icc)
            print("Total initial capital cost        %.2f M USD" % (icc * n_turbine * t_rating * 1.0e-006))
            print("Opex costs of the park            %.2f M USD/yr" % (c_opex_turbine * n_turbine * 1.0e-006))
            print("Net energy capture                %.2f MWh/MW/yr" % nec)
            print(
                "LCoE                              %.2f USD/MWh" % (lcoe * 1.0e003)
            )  # removed "coe", best to have only one metric for cost
            print("################################################")

    def compute_partials(self, inputs, J, discrete_inputs):
        J = self.J
