import numpy as np
class Tratio():
    def __init__(self, parameters, sampler, config, parser):
        required_sampled_parameters = ['box-kymin', 'units-bref', 'units-nref', 'units-tref', 'nongene-Ti/Te', '_grp_species_1-omt', 'geometry-q0', 'species-omn']
        if parameters != required_sampled_parameters:
            raise NotImplementedError(f'Daniel Says: This class is intended for a specific use and should only be used if the parameters being sampled are these\n{required_sampled_parameters}')
        
        # species 0 is ions, species 1 is electrons, since in base parameters the ions are first
        gene_parameters = ['box-kymin', 'units-bref', 'units-nref', 'units-tref', 
                           'geometry-q0', 
                           '_grp_species_0-omn', '_grp_species_1-omn',
                           '_grp_species_0-omt','_grp_species_1-omt',
                           '_grp_species_0-temp',
                           ]# Tau = (Ti/Te)^-1, try setting tau to -1 first.
        # when making gene_samples care needs to be taken to use the correct units from the gene docs
        gene_samples = {}
        gene_samples['box-kymin'] = sampler.samples['box-kymin']
        gene_samples['units-bref'] = sampler.samples['units-bref']
        gene_samples['units-nref'] = sampler.samples['units-nref']
        gene_samples['units-tref'] = sampler.samples['units-tref']
        gene_samples['geometry-q0'] = sampler.samples['geometry-q0']
        gene_samples['species-omn'] = sampler.samples['species-omn'] # this is converted later into '_grp_species_0-omn', '_grp_species_1-omn',
        gene_samples['_grp_species_1-omt'] = sampler.samples['_grp_species_1-omt']
        
        Ti_Te = np.array(sampler.samples['nongene-Ti/Te'])
        Te = np.array(sampler.samples['units-tref'])
        Ti = (Ti_Te * Te)
        gene_samples['_grp_species_0-temp'] = Ti/Te
        
        Lref = parser.get_parameter_value(config.base_params_path, group_var = ['units', 'lref'])
        
        # Convert to un-normalised gradient
        grad_Te = -sampler.samples['_grp_species_1-omt'] * (Te / Lref)

        Te_0 = Te
        Te_1 = grad_Te + Te_0
        Ti_0 = Ti
        Ti_1 = Te_1 * Ti_Te
        grad_Ti = (Ti_1 - Ti_0)/1 # rise over run

        omti = -(Lref/Te) * grad_Ti
        gene_samples['_grp_species_0-omt'] = omti

        self.samples = gene_samples



        


