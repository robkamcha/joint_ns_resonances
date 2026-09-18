import numpy as np
import matplotlib.pyplot as plt
import copy

from bilby.gw.source import lal_binary_neutron_star
from bilby.gw.likelihood.relative import RelativeBinningGravitationalWaveTransient
import bilby

from tbilby.core.prior.order_stats import TransdimensionalConditionalAscendingOrderStatPrior
import tbilby


logger = bilby.core.utils.logger
outdir = "BBH_quick_test"
label = "test_run"
sampling_frequency = 2048.0
trigger_time = 1384782888.6
detectors = ["H1", "L1", "V1"]
maximum_frequency = sampling_frequency // 2
minimum_frequency = 20
roll_off = 0.4
duration = 8 # s
post_trigger_duration = 5 #s
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time


parameter_dict = {
    'mass_1': 30.0,
    'mass_2': 28.0,
    'a_1': 0.02,
    'a_2': 0.01,
    'tilt_1': 0.0,
    'tilt_2': 0.0,
    'phi_12': 0.0,
    'phi_jl': 0.0,
    'luminosity_distance': 300,  # Mpc
    'theta_jn': np.pi / 3.0,
    'phase': 0.0,
    'geocent_time': trigger_time,
    'psi': 0.0,
    'ra':1.375,
    'dec':-1.2108,
}

mc_injected = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1=parameter_dict['mass_1'], mass_2=parameter_dict['mass_2'])
q_injected = bilby.gw.conversion.component_masses_to_mass_ratio(mass_1=parameter_dict['mass_1'], mass_2=parameter_dict['mass_2'])

parameter_dict['chirp_mass'] = mc_injected
parameter_dict['mass_ratio'] = q_injected

# ── Waveform generators ────────────────────────────────────────────────────────
waveform_arguments = dict(
    minimum_frequency=minimum_frequency,
    maximum_frequency=maximum_frequency,
    reference_frequency=minimum_frequency,
    waveform_approximant="IMRPhenomXAS",
)
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)


# ── Setup ifos ───────────────────────────────────────────────────────────────
ifo_list = bilby.gw.detector.InterferometerList(detectors)
ifo_list.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)

print("Injecting signal...")
ifo_list.inject_signal(
    waveform_generator=waveform_generator, 
    parameters=parameter_dict,
    earth_rotation=True
)

def get_network_snr(ifos, wf_gen, parameters):
    ifos = copy.deepcopy(ifos) # Don't modify the original ifos
    network_snr = 0.0
    injection_polarisations = wf_gen.frequency_domain_strain(parameters)

    for ifo in ifos:
        ifo_polarisations = ifo.get_detector_response(injection_polarisations, parameters, earth_rotation=True)
        ifo_snr_squared = ifo.optimal_snr_squared(ifo_polarisations)
        network_snr += ifo_snr_squared

    return np.sqrt(network_snr)

network_snr = get_network_snr(ifo_list, waveform_generator, parameter_dict)
print(f"Network SNR: {network_snr:.2f}")


# ── Likelihood ─────────────────────────────────────────────────────────────────
print("Setting up likelihood...")
likelihood =  bilby.gw.GravitationalWaveTransient(
    interferometers=ifo_list,
    waveform_generator=waveform_generator,
)

# ── Priors ─────────────────────────────────────────────────────────────────────
priors = bilby.core.prior.dict.PriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
    "theta_jn"
]:
    priors[key] = parameter_dict[key]

priors['chirp_mass'] = bilby.core.prior.Uniform(minimum=10, maximum=60, name="chirp_mass", latex_label='$\\mathcal{M}$', unit='$M_\\odot$')
priors['mass_ratio'] = bilby.core.prior.Uniform(minimum=0.5, maximum=1, name="mass_ratio", latex_label='$q$')
priors['luminosity_distance'] = bilby.core.prior.Uniform(minimum=100, maximum=600, name="d_L", latex_label='$d_L$', unit='Mpc')


# ── Sampling ───────────────────────────────────────────────────────────────────
print("Sampling...")
result = bilby.core.sampler.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    sample="rwalk",
    nlive=1000,
    nact=80,
    outdir=outdir,
    label=label,
    resume=True,
    npool=16,
)

