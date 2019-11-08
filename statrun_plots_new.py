import matplotlib.pyplot as plt
import numpy as np
import pickle
from plot_tools import nice_plot_colors
from cycler import cycler

plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)
lcolors = nice_plot_colors.vlines

markerlist = ['None','|','o','v','d','^']
dashlist = [(3,1),(4,1),(3,1),(None,None),(None,None),(None,None)]
plt.rc('lines', linewidth=1.5, markersize=3.5, markeredgewidth=0.0)
plt.rc('axes', prop_cycle=(cycler('color', lcolors)
                           +cycler('linestyle', ['--','--','--','-','-','-'])
                           +cycler('dashes', dashlist)
                           # +cycler('marker', markerlist)
                           #+cycler('mfc',  nice_plot_colors.vbars)
                           +cycler('linewidth', [1.0, 2.0, 1.0, 2.0,1.0, 2.0]) 
                           ))

def load_and_plot(DATA_DIR, nowstr, *args, **kwargs):
    with open(DATA_DIR+nowstr+'.p', "rb") as fh:
        best_path_cost = pickle.load(fh)    
        true_path_cost = pickle.load(fh)    
        est_path_cost = pickle.load(fh)    
        est_path_var = pickle.load(fh)
        sample_time = pickle.load(fh)
    return make_plots(best_path_cost, true_path_cost, est_path_cost, sample_time, *args, **kwargs)
    
def append_data(DATA_DIR, nowstrs, outfile):
    nowstr = nowstrs[0]
    with open(DATA_DIR+nowstr+'.p', "rb") as fh:
        best_path_cost = pickle.load(fh)    
        true_path_cost = pickle.load(fh)    
        est_path_cost = pickle.load(fh)    
        est_path_var = pickle.load(fh)
        sample_time = pickle.load(fh) 
    for nowstr in nowstrs[1:]:
        with open(DATA_DIR+nowstr+'.p', "rb") as fh:
            best_path_cost = np.hstack((best_path_cost, pickle.load(fh)))
            true_path_cost = np.vstack((true_path_cost, pickle.load(fh)))   
            est_path_cost = np.vstack((est_path_cost, pickle.load(fh)))
            est_path_var = np.vstack((est_path_var, pickle.load(fh)))
            sample_time = np.vstack((sample_time, pickle.load(fh)))
    with open(DATA_DIR+outfile+'.p', "wb") as fh:
        pickle.dump(best_path_cost, fh)
        pickle.dump(true_path_cost, fh)
        pickle.dump(est_path_cost, fh)
        pickle.dump(est_path_var, fh)
        pickle.dump(sample_time, fh)

def save_plots(FIG_DIR, nowstr, figs=None, DATA_DIR=None, *args, **kwargs):
    if figs == None:
        figs = load_and_plot(DATA_DIR, nowstr, *args, **kwargs)
    fletters = ['C','','L','E','V','T']
    for fl,fig in zip(fletters,figs):
        fig.savefig(FIG_DIR+nowstr+fl+'.pdf', bbox_inches='tight')
    return figs

def make_plots(best_path_cost, true_path_cost, est_path_cost, sample_time, labels, cols=None, comparison=3, n_bars = 10, ls_cycle=False, fig_size=[5,4],bar_width = 0.2):
    # Input arrays are [numruns, nummethods, numsamples]
    NUM_STATRUNS = true_path_cost.shape[0]
    nmethods = true_path_cost.shape[1]
    NUM_SAMPLES = true_path_cost.shape[2]
    if cols == None:
        cols = lcolors
        #cmap = plt.get_cmap('jet')
        #cols = cmap(np.linspace(0, 1.0, nmethods))
    if ls_cycle == True:
        lss = ['-','--','-.',':']
    else:
        lss = ['-']
    xlabel = r'Number of observations, $t$'
    
    fig1, ax1 = plt.subplots()
    path_est_error = (true_path_cost - est_path_cost)/true_path_cost
    fig1.set_size_inches(fig_size[0], fig_size[1])
    
    for i in range(nmethods):
        RMS = np.zeros(NUM_SAMPLES, dtype='float')
        for j in range(NUM_SAMPLES):
            RMS[j] = np.sqrt(np.mean(np.power([k for k in path_est_error[:,i,j] if not np.isnan(k)], 2)))
        #RMS = np.sqrt(np.mean(np.power(path_est_error[:,i,:], 2), axis=0))
        ax1.plot(np.arange(NUM_SAMPLES)+1, RMS, color=cols[i%len(cols)], ls=lss[i%len(lss)], label=labels[i])
    ax1.set_xlim(0, NUM_SAMPLES)
    ax1.legend(loc=0, prop={'size':10})
    ax1.set_xlabel(xlabel)
    ax1.grid(True, which='major', axis='y')
    ax1.set_ylabel('Normalized path prediction RMSE')
    
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(fig_size[0], fig_size[1])
    xticks = np.arange(0, NUM_SAMPLES+1, NUM_SAMPLES/n_bars)
    bar_positions = xticks[1:]
    for i in range(nmethods):
        tcol,ls,dsh = cols[i%len(cols)],lss[i%len(lss)],dashlist[i%len(dashlist)]
        tempdata = [np.divide(true_path_cost[:,i,j], best_path_cost)-1 for j in bar_positions-1]
        tempdata = [td[~np.isnan(td)] for td in tempdata]
        pos = bar_positions+bar_width*(i-nmethods/2)
        ax2.boxplot(tempdata, positions=pos, showbox=True, notch=True, boxprops={'color':tcol, 'facecolor':tcol }, 
            medianprops={'color':tcol}, 
            showcaps=True, capprops={'color':tcol,'marker':None},
            whis=0, showfliers=False, 
            whiskerprops={'color':tcol}, flierprops={'color':tcol}, 
            bootstrap=5000, patch_artist=True, widths=bar_width*.6)  #, 

        tempdata = [np.divide(true_path_cost[:,i,j], best_path_cost)-1 for j in range(0, NUM_SAMPLES)] 
        ax2.plot(np.arange(1,NUM_SAMPLES+1), [np.median(td[~np.isnan(td)]) for td in tempdata], color=tcol, ls=ls, dashes=dsh, label=labels[i])
    ax2.set_xlim(0, NUM_SAMPLES+1)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks)
    ax2.set_yscale('log')
    pc_labels = [r'${0:0.1f}\%$'.format(yval*100.0) for yval in ax2.get_yticks()]
    ax2.set_yticklabels(pc_labels)
    ax2.grid(True, which='major', axis='y')
    ax2.legend(loc=0, prop={'size':10})
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r'Additional path cost (vs. best path)')
    ax2.set_ylim(0.0005, 0.5)

    
    fig3, ax3 = plt.subplots()
    fig3.set_size_inches(fig_size[0], fig_size[1])
    between_samples_gap = 0.2
    columns_ratio = 1.0
    columns_width = columns_ratio*(1-between_samples_gap)/nmethods
    lowcost_method = np.zeros((nmethods,NUM_SAMPLES))
    for jj in range(NUM_SAMPLES):
        for kk in range(NUM_STATRUNS):
            lowcost_method[np.argmin(true_path_cost[kk,:,jj]), jj] += 1
    for jj in range(nmethods):
        #start_pos = 0.5+between_samples_gap/2+jj*(1-between_samples_gap)/nmethods
        #ax3.bar(np.arange(start_pos, NUM_SAMPLES+0.49), lowcost_method.transpose()[:,jj]/NUM_STATRUNS*100.0, columns_width, color=cols[jj], label=labels[jj])
        ax3.plot(np.arange(NUM_SAMPLES)+1, lowcost_method.transpose()[:,jj]/NUM_STATRUNS*100.0, color=cols[jj%len(cols)], ls=lss[jj%len(lss)], label=labels[jj])
    ax3.set_xlim(0, NUM_SAMPLES+1)
    ax3.grid(True, which='major', axis='y')
    ax3.legend(loc=0, prop={'size':10})
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel('Best method (\%)')    
    
    
    fig4, ax4 = plt.subplots()
    fig4.set_size_inches(fig_size[0], fig_size[1])
    comp_methods  = set(range(nmethods))
    comp_methods.remove(comparison)
    for i in comp_methods:
        tempdata = np.divide(true_path_cost[:,i,:], true_path_cost[:,comparison,:])*100.0
        ax4.plot(np.arange(1,NUM_SAMPLES+1), [np.median(td[~np.isnan(td)]) for td in tempdata.transpose()], color=cols[i%len(cols)], ls=lss[i%len(lss)], label=labels[i])
        
        tempdata = tempdata[:,bar_positions-1]
        pos = bar_positions+bar_width*(i-nmethods/2)
        
        #stds = np.squeeze(np.std(tempdata, axis=0))
        #ax4.errorbar(pos, np.squeeze(np.mean(tempdata, axis=0)), stds, lw=3)
       
        for k,td in enumerate(tempdata.transpose()):
            tcol = cols[i%len(cols)]
            ax4.boxplot(td[~np.isnan(td)] , positions=[pos[k]], showbox=True, notch=True, showcaps=False, whis=0, showfliers=False, 
            boxprops={'color':tcol, 'facecolor':tcol }, whiskerprops={'color':tcol}, flierprops={'color':tcol}, 
            bootstrap=5000, patch_artist=True)  #, , medianprops={'color':cols[i]}
    ax4.set_xlim(0, NUM_SAMPLES+1)
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(xticks)
    ax4.grid(True, which='major', axis='y')
    ax4.legend(loc=0, prop={'size':10})
    ax4.set_xlabel(xlabel)
    # ax4.set_title('Path cost error relative to {0} method'.format(labels[comparison]))
    ax4.set_ylabel('Mean path cost relative to {0} (\%)'.format(labels[comparison]))  
    
    fig5, ax5 = plt.subplots()
    fig5.set_size_inches(fig_size[0], fig_size[1])
    percent_win = np.zeros((nmethods,NUM_SAMPLES))
    for jj in comp_methods:
        for kk in range(NUM_SAMPLES):
            percent_win[jj,kk] = (true_path_cost[:,jj,kk] < true_path_cost[:,comparison,kk]).sum()
        ax5.plot(np.arange(NUM_SAMPLES)+1, percent_win[jj,:]/NUM_STATRUNS*100.0, color=cols[jj%len(cols)],ls = lss[jj%len(lss)], label=labels[jj])
    ax5.set_xlim(0, NUM_SAMPLES+1)
    ax5.grid(True, which='major', axis='y')
    ax5.legend(loc=0, prop={'size':10})
    ax5.set_xlabel(xlabel)
    ax5.set_ylabel('Lower path cost vs {0} (\%)'.format(labels[comparison]))
    
    fig6, ax6 = plt.subplots()
    fig6.set_size_inches(fig_size[0], fig_size[1])
    tsummary = [sample_time[:,i,:].ravel() for i in range(nmethods)]
    ax6.boxplot(tsummary)
    ax6.set_xticklabels(labels)
    ax6.set_ylabel('Sample time (s)')
    
    return fig1, fig2, fig3, fig4, fig5, fig6
    
    

    #lowcost_rand = lowcost_method.transpose()[:,0]/NUM_STATRUNS*100;
    #lowcost_maxv = lowcost_method.transpose()[:,1]/NUM_STATRUNS*100;
    #lowcost_fm = lowcost_method.transpose()[:,2]/NUM_STATRUNS*100;
    #h_lowcost = [ax3.bar(np.arange(0.75, NUM_SAMPLES), lowcost_rand, 0.5, color=cols[0], label=labels[0])]
    #h_lowcost.append(ax3.bar(np.arange(0.75, NUM_SAMPLES), lowcost_maxv, 0.5, bottom=lowcost_rand, color=cols[1], label=labels[1]))
    #h_lowcost.append(ax3.bar(np.arange(0.75, NUM_SAMPLES), lowcost_fm, 0.5, bottom=(lowcost_rand+lowcost_maxv), color=cols[2], label=labels[2]))



    #fig4, ax4 = plt.subplots()
    #fig4.set_size_inches(fig_size[0], fig_size[1])
    #mcost = []
    #for i in range(nmethods):
    #    tempdata = [np.divide(true_path_cost[:,i,j], best_path_cost)-1 for j in range(0, NUM_SAMPLES)]
    #    mcost.append(np.squeeze(np.nanmean(tempdata, axis=2)))
    #
    #a  = set(range(nmethods))
    #a.remove(comparison)
    #for i in a:
    #    tempdata = np.divide(mcost[i], mcost[comparison])*100
    #    ax4.plot(np.arange(NUM_SAMPLES)+1, tempdata, color=cols[i], label=labels[i])
    #ax4.set_xlim(0, NUM_SAMPLES+1)
    #ax4.grid(True, which='major', axis='y')
    #ax4.legend(loc=0, prop={'size':10})
    #ax4.set_xlabel(xlabel)
    ## ax4.set_title('Path cost error relative to {0} method'.format(labels[comparison]))
    #ax4.set_ylabel('Relative path cost error (\%)'.format(labels[comparison]))  