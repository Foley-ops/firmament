"""Physical constants and material properties (SI). Real values, simplified layers."""
SIGMA = 5.670374419e-8      # Stefan–Boltzmann, W/m^2 K^4
GREENHOUSE = 0.78           # fraction of surface longwave absorbed by air (Earth-like ~0.77-0.8)
ALBEDO_ROCK = 0.30
ALBEDO_WATER = 0.06
K_WATER_LIGHT = 0.3         # Beer–Lambert attenuation in water, 1/m (clear water PAR)
C_AIR = 1.0e6               # J/m^2 K — boundary-layer air column (~100 m; full column is 1e7)
C_SURF_DRY = 6.5e5          # J/m^2 K — top ~0.3 m rock (rho 2700, c 800)
C_SED = 4.3e6               # J/m^2 K — 2 m sediment/rock
CW_VOL = 4.186e6            # J/m^3 K — volumetric heat capacity of water
H_AIR_SURF = 15.0           # W/m^2 K sensible exchange
H_SURF_SED = 3.0            # W/m^2 K conduction across interface
K_HORIZ = 2.0               # W/K per face — lateral conduction (rock k~2 W/mK, 1 m cell)
LV = 2.5e6                  # J/kg latent heat of vaporization
CW_SP = 4186.0              # J/kg K specific heat of water
T_REF = 288.0               # K reference temperature for vapor enthalpy bookkeeping
G = 9.81                    # m/s^2
RHO_W = 1000.0              # kg/m^3
