import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as int
import pandas as pd

def init_GASLIB():
    # Label    "Gas Name"   "Mwt"
    # Unit     "-"          "kg/kmol" 
    column = [ "Name",      "Mwt"]   
    gaslib = [["Hydrogen",  2], \
              ["Methane",   16], \
              ["Ethane",    30], \
              ["Propane",   44], \
              ["n-Butane",  58], \
              ["i-Butane",  58], \
              ["CarbonDiox",44], \
              ["Nitrogen",  28], \
              ["R22",       86.47]]
    index = [row[0] for row in gaslib]
    GASLIB = pd.DataFrame(gaslib, columns=column, index=index)
    return GASLIB


def init_PARLIB():
    column = [
         "Symbol",  "Name",             "Unit",             "Value"
    ]
    params = [
        ["R",       "GasConst",         "m2-kg/s2/K/mol",   8.3144598], \
        ["kb",      "BoltzmanConst",    "m2-kg/s2/K",       1.3806485e-23], \
        ["Patm",    "ATM pressure",     "MPaA",             0.101325], \
        ["Tatm",    "0 degC",           "K",                273.15]
        ]
    PARLIB = pd.DataFrame(params, columns=column)
    PARLIB = PARLIB.set_axis(PARLIB.Symbol, axis=0)
    return PARLIB


def calc_V_EOS_mol(P_MPaA, T_K, m_mol):
    """_V=mRT/P with m : mol_
    Solve Equation of State(EOS) for V [m3]

    Args:
        P_MPaA (_float_)    : _Pressure     unit : MPaA_
        T_K (_float_)       : _Temperature  unit : K_
        m_mol (_float_)     : _Mol          unit : mol_

    Returns:
        _float_: _Volume    unit : m3_
    """
    V_m3 = (10**(-6)) * m_mol * PARLIB.loc["R"]["Value"] * T_K / P_MPaA
    return V_m3


def calc_P_EOS_mol(V_m3, T_K, m_mol):
    """_P=mRT/V with m : mol_
    Solve Equation of State(EOS) for P [MPaA] 

    Args:
        V_m3 (_float_)      : _Volume       unit : m3_
        T_K (_float_)       : _Temperature  unit : K_ 
        m_mol (_float_)     : _Mol          unit : mol_

    Returns:
        _float_: _Pressure  unit : MPaA_
    """
    P_MPaA = (10**(-6)) * m_mol * PARLIB.loc["R"]["Value"] * T_K / V_m3
    return P_MPaA


def calc_T_EOS_mol(P_MPaA, V_m3, m_mol):
    """_T=PV/m/R with m : mol_
    Solve Equation of State(EOS) for T [K]

    Args:
        P_MPaA (_float_)    : _Pressure     unit : MPaA_
        V_m3 (_float_)      : _Volumr       unit : m3_
        m_mol (_float_)     : _Mol          unit : mol_

    Returns:
        _float_: _Temperature   unit : K_
    """
    T_K = (10**6) * P_MPaA * V_m3 / m_mol / PARLIB.loc["R"]["Value"]
    return T_K


def calc_P_CGL(P1, V1, T1, V2, T2):
    """_P2=P1*V1/V2*T2/T1_
    Solve Combined Gas Low(CGL) for P2

    Args:
        P1 (_float_): _Pressure at state 1_
        V1 (_float_): _Volume at state 1_
        T1 (_float_): _Temperature at state 1_
        V2 (_float_): _Volume at state 2_
        T2 (_float_): _Temperature at state 2_

    Returns:
        _float_: _Pressure at state 2_
    """
    P2 = P1 * (V1 / V2) * (T2 / T1)
    return P2


def calc_V_CGL(P1, V1, T1, P2, T2):
    """_V2=V1*P1/P2*T2/T1_
    Solve Combined Gas Low(CGL) for V2

    Args:
        P1 (_float_): _Pressure at state 1_
        V1 (_float_): _Volume at state 1_
        T1 (_float_): _Temperature at state 1_
        P2 (_float_): _Pressure at state 2_
        T2 (_float_): _Temperature at state 2_

    Returns:
        _float_: _Volume at state 2_
    """
    V2 = V1 * (P1 / P2) * (T2 / T1)
    return V2


def calc_T_CGL(P1, V1, T1, P2, V2):
    """_T2=T1*P2/P1*V2/V1_
    Solve Combined Gas Low(CGL) for T2

    Args:
        P1 (_float_): _Pressure at state 1_
        V1 (_float_): _Volume at state 1_
        T1 (_float_): _Temperature at state 1_
        P2 (_float_): _Pressure at state 2_
        V2 (_float_): _Volume at state 2_

    Returns:
        _float_: _Temperature at state 2_
    """
    T2 = T1 * (P2 / P1) * (V2 / V1)
    return T2


def calc_V_Normal(P_MPaA, V_m3, T_K):
    """_VN=V*P/PN*TN/T_
    Calculate Volume at Normal Condition, definition of Normal is NTP
        TN = 0 degC = 273.15 K
        PN = 0.101325 MPaA

    Args:
        P_MPaA (_float_): _Pressure     unit : MPaA_
        V_m3 (_float_): _Volume         unit : m3_
        T_K (_float_): _Temperature     unit : K_

    Returns:
        _float_: _Volume    unit : Nm3_
    """
    V_Nm3 = V_m3 * (P_MPaA / PARLIB.loc["Patm"]["Value"]) * (PARLIB.loc["Tatm"]["Value"] / T_K)
    return V_Nm3


def calc_M_Normal(V_Nm3, Mwt):
    """_M=Mwt*V/22.4_
    Calculate Mass of Gass

    Args:
        V_Nm3 (_float_): _Volume    unit : Nm3_
        Mwt (_float_): _Mole Weight unit : kg/kmol_

    Returns:
        _float_: _Mass  unit : kg_
    """
    M_kg = Mwt * V_Nm3 / 22.4
    return M_kg


def calc_V_EOS_kg(P_MPaA, T_K, M_kg, Mwt):
    """_V=MRT/P with M : kg_
    Solve Equation of State(EOS) for V [m3]

    Args:
        P_MPaA (_float_): _Pressure     unit : MPaA_
        T_K (_float_): _Temperature     unit : K_
        M_kg (_float_): _Mass           unit : kg_
        Mwt (_float_): _Mole Weight     unit : g/mol_

    Returns:
        _float_: _Volumr    unit : m3_
    """
    R_M = PARLIB.loc["R"]["Value"] / Mwt * 1000
    V_m3 = (10**(-6)) * M_kg * R_M * T_K / P_MPaA
    return V_m3


def calc_P_EOS_kg(V_m3, T_K, M_kg, Mwt):
    """_P=MRT/V with M : kg_
    Solve Equation of State(EOS) for P [MPaA]

    Args:
        V_m3 (_float_): _Volume         unit : m3_
        T_K (_float_): _Temperature     unit : K_
        M_kg (_float_): _Mass           unit : kg_
        Mwt (_float_): _Mole Weight     unit : g/mol_

    Returns:
        _float_: _Pressure    unit : MPaA_
    """
    R_M = PARLIB.loc["R"]["Value"] / Mwt * 1000
    P_MPaA = (10**(-6)) * M_kg * R_M * T_K / V_m3
    return P_MPaA


def calc_T_EOS_kg(P_MPaA, V_m3, M_kg, Mwt):
    """_T=PV/MR with M : kg_
    Solve Equation of State(EOS) for T [K]

    Args:
        P_MPaA (_float_): _Pressure     unit : MPaA
        V_m3 (_float_): _Volume         unit : m3_
        M_kg (_float_): _Mass           unit : kg_
        Mwt (_float_): _Mole Weight     unit : g/mol_

    Returns:
        _float_: _Temperature   unit : K_
    """
    R_M = PARLIB.loc["R"]["Value"] / Mwt * 1000
    T_K = (10**6) * P_MPaA * V_m3 / M_kg / R_M
    return T_K


def calc_P_ADC(P1, V1, V2, Kappa):
    """_P2=P1*(V1/V2)^Kappa_
    Solve Adiabatic Change(ADC) for Pressure

    Args:
        P1 (_float_): _description_
        V1 (_float_): _description_
        V2 (_float_): _description_
        Kappa (_float_): _description_

    Returns:
        _type_: _description_
    """
    P2 = P1 * (V1 / V2) ** Kappa
    return Kappa


def calc_lt_ideal_ADC(P1_MPaA, P2_MPaA, v1_m3, Kappa):
    """_Calculate ideal Technical Worl lt at Adiabatic Compression(ADC)_
    ideal means no over/under compression

    Args:
        P1_MPaA (_float_): _Pressure at state 1 unit : MPaA_
        P2_MPaA (_float_): _Pressure at state 2 unit : MPaA_
        v1_m3 (_float_): _Specific Volume       unit : m3/kg_
        Kappa (_float_): _Specific Heat ratio   unit : -_

    Returns:
        _float_: _Specific Technical Work   unit : J/kg_
    """
    P1_PaA = (10**6) * P1_MPaA
    P2_PaA = (10**6) * P2_MPaA
    const = P1_PaA * (v1_m3 ** Kappa)
    v_adc = lambda p: p ** (-1/Kappa)
    lt_J = (const ** (1/Kappa)) * int.quad(v_adc, P1_PaA, P2_PaA)[0]
    return lt_J


def calc_lt_vi_ADC(P1_MPaA, P2_MPaA, v1_m3, vi, Kappa):
    """_Calculate Technical Worl lt at Adiabatic Compression(ADC)_
    This calculation considering compressor internal vi and over/under compression
    Physically each value means
        P1  :   Compressor Suction Pressure
        P2  :   Compressor Discharge Pressure
        PI  :   Compressor Internal max Pressure(Ps*vi^K)
        v1  :   Specific Volume at top of volume curve

    Args:
        P1_MPaA (_float_): _Pressure at State 1     unit : MPaA_
        P2_MPaA (_float_): _Pressure at State 2     unit : MpaA_
        v1_m3 (_float_): _Specific Volume           unit : m3/kg_
        vi (_float_): _Compressor internal vi       unit : -_
        Kappa (_float_): _Specific Heat Ratio       unit : -_

    Returns:
        _float_: _Specific Technical Work   unit : J/kg_
    """
    P1_PaA = (10**6) * P1_MPaA  # ~ Suction Pressure
    P2_PaA = (10**6) * P2_MPaA  # ~ Discharge Pressure
    PI_PaA = P1_PaA * (vi ** Kappa)
    const = P1_PaA * (v1_m3 ** Kappa)
    v2_m3 = v1_m3 / vi
    v_adc = lambda p: p ** (-1/Kappa)
    lt_J_vi = (const ** (1/Kappa)) * int.quad(v_adc, P1_PaA, PI_PaA)[0]
    lt_J_uo = v2_m3 * (P2_PaA - PI_PaA)
    # Under compression :   lt_J_uo > 0
    # Over  compression :   lt_J_uo < 0
    lt_J = lt_J_vi + lt_J_uo
    return lt_J



class RefCycle():
    def __init__(self, refrigerant):
        self.PARLIB = init_PARLIB()
        self.GAS = init_GASLIB().loc[refrigerant]


