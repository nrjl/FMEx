import statrun_plots_new
import matplotlib.pyplot as plt

FMEX_DIR = '../'
DATA_DIR = FMEX_DIR+'data/'
VID_DIR = FMEX_DIR+'vid/'
FIG_DIR = FMEX_DIR+'fig/'



#n_fmexbase_samples = 10
#labels = ['FMEx $1\sigma$ ({0})'.format(n_fmexbase_samples), 
#    'FMEx $1\sigma$ ({0})'.format(3*n_fmexbase_samples), 
#    'FMEx $1\sigma$ ({0})'.format(6*n_fmexbase_samples), 
#    'FMEx $3\sigma$ ({0})'.format(n_fmexbase_samples), 
#    'FMEx $3\sigma$ ({0})'.format(3*n_fmexbase_samples),  
#    'FMEx $3\sigma$ ({0})'.format(6*n_fmexbase_samples)]
labels = ['Random (1)', 
    'Max Var (10000)', 
    'LCB (10000)',
    'FMEx $1\sigma$ (150)', 
    'FMEx $3\sigma$ (50)', 
    'FMEx MC6 (50)']
mkwargs = {'labels':labels, 'comparison':1, 'n_bars':8, 'fig_size':[7,3.5], 'bar_width':0.3}

statrun_plots_new.load_and_plot(DATA_DIR, 'noobs_collated', **mkwargs)
plt.show()


#statrun_plots_new.save_plots(FIG_DIR, 'obs_collated', DATA_DIR=DATA_DIR, **mkwargs); plt.close('all')
#statrun_plots_new.save_plots(FIG_DIR, 'noobs_collated', DATA_DIR=DATA_DIR, **mkwargs); plt.close('all')
mkwargs['fig_size'] = [4,2.5]
statrun_plots_new.save_plots(FIG_DIR, 'obs_collated', DATA_DIR=DATA_DIR, **mkwargs); plt.close('all')
statrun_plots_new.save_plots(FIG_DIR, 'noobs_collated', DATA_DIR=DATA_DIR, **mkwargs); plt.close('all')
