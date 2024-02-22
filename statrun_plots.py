import matplotlib.pyplot as plt
import numpy as np
import pickle
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

def load_and_plot(DATA_DIR, nowstr, labels = ['Random', 'Max Variance', 'LCB', 'FMEx']):
    fh = open(DATA_DIR+nowstr+'.p', "rb")
    best_path_cost = pickle.load(fh)    
    true_path_cost = pickle.load(fh)    
    est_path_cost = pickle.load(fh)    
    est_path_var = pickle.load(fh)
    fh.close()
    fig1, fig2, fig3, fig4 = make_plots(best_path_cost, true_path_cost, est_path_cost, labels, comparison=3)
    return fig1,fig2,fig3,fig4   

def save_plots(FIG_DIR, nowstr, fig1,fig2,fig3,fig4):
    fig1.savefig(FIG_DIR+nowstr+'C.pdf', bbox_inches='tight')
    fig2.savefig(FIG_DIR+nowstr+'.pdf', bbox_inches='tight')
    fig3.savefig(FIG_DIR+nowstr+'L.pdf', bbox_inches='tight')
    fig4.savefig(FIG_DIR+nowstr+'E.pdf', bbox_inches='tight')

def make_plots(best_path_cost, true_path_cost, est_path_cost, labels, cols=None, comparison=3):
    fig_size = 4
    # Input arrays are [numruns, nummethods, numsamples]
    NUM_STATRUNS = true_path_cost.shape[0]
    nmethods = true_path_cost.shape[1]
    NUM_SAMPLES = true_path_cost.shape[2]
    if cols == None:
        cmap = plt.get_cmap('jet')
        cols = cmap(np.linspace(0, 1.0, nmethods))
    
    fig1, ax1 = plt.subplots()
    path_est_error = (true_path_cost - est_path_cost)/true_path_cost
    fig1.set_size_inches(fig_size+1, fig_size)
    
    for i in range(nmethods):
        RMS = np.zeros(NUM_SAMPLES, dtype='float')
        for j in range(NUM_SAMPLES):
            RMS[j] = np.sqrt(np.mean(np.power([k for k in path_est_error[:,i,j] if not np.isnan(k)], 2)))
        #RMS = np.sqrt(np.mean(np.power(path_est_error[:,i,:], 2), axis=0))
        ax1.plot(np.arange(NUM_SAMPLES)+1, RMS, color=cols[i%len(cols)], label=labels[i])
    ax1.set_xlim(0, NUM_SAMPLES)
    ax1.legend(loc=0, prop={'size':10})
    ax1.set_xlabel('Number of samples')
    ax1.grid(True, which='major', axis='y')
    ax1.set_ylabel('Normalised path prediction RMS error')    
    
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(fig_size+1, fig_size)
    sample_bars = int(NUM_SAMPLES/10)
    for i in range(nmethods):
        tempdata = [np.divide(true_path_cost[:,i,j], best_path_cost)-1 for j in np.arange(0, NUM_SAMPLES, sample_bars)]
        tempdata = [td[~np.isnan(td)] for td in tempdata]
        pos = np.arange(0,NUM_SAMPLES*nmethods,sample_bars*nmethods)+i+1
        ax2.boxplot(tempdata, positions=pos, showbox=True, notch=True, showcaps=False, whis=0, showfliers=False, 
            boxprops={'color':cols[i]}, whiskerprops={'color':cols[i]}, flierprops={'color':cols[i]}, 
            bootstrap=5000)  #, , medianprops={'color':cols[i]}
        tempdata = [np.divide(true_path_cost[:,i,j], best_path_cost)-1 for j in range(0, NUM_SAMPLES)] 
        ax2.plot(np.arange(0,NUM_SAMPLES*nmethods,nmethods)+i+1, [np.median(td[~np.isnan(td)]) for td in tempdata], cols[i], label=labels[i])
    ax2.set_xlim(0, nmethods*NUM_SAMPLES)
    ax2.set_xticks(np.arange(0,NUM_SAMPLES*nmethods,sample_bars*nmethods)+2)
    ax2.set_xticklabels(list(range(1,NUM_SAMPLES+1, sample_bars)))
    ax2.set_yscale('log')
    ax2.grid(True, which='major', axis='y')
    ax2.legend(loc=0, prop={'size':10})
    ax2.set_xlabel('Number of samples')
    ax2.set_ylabel('Additional path cost (normalised against best path)')
    #ax2.set_ylim(8e-3, 1e0)

    
    fig3, ax3 = plt.subplots()
    fig3.set_size_inches(fig_size+1, fig_size)
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
        ax3.plot(np.arange(NUM_SAMPLES)+1, lowcost_method.transpose()[:,jj]/NUM_STATRUNS*100.0, color=cols[jj], label=labels[jj])
    ax3.set_xlim(0, NUM_SAMPLES+1)
    ax3.grid(True, which='major', axis='y')
    ax3.legend(loc=0, prop={'size':10})
    ax3.set_xlabel('Number of samples')
    ax3.set_ylabel('Best method (\%)')    
    
    
    fig4, ax4 = plt.subplots()
    fig4.set_size_inches(fig_size+1, fig_size)
    comp_methods  = set(range(nmethods))
    comp_methods.remove(comparison)
    for i in comp_methods:
        tempdata = np.divide(true_path_cost[:,i,:], true_path_cost[:,comparison,:])*100.0
        ax4.plot(np.arange(0,NUM_SAMPLES*nmethods,nmethods)+i+1, [np.median(td[~np.isnan(td)]) for td in tempdata.transpose()], color=cols[i], label=labels[i])
        
        tempdata = tempdata[:,np.arange(0, NUM_SAMPLES, sample_bars)]
        pos = np.arange(0,NUM_SAMPLES*nmethods,sample_bars*nmethods)+i+1
        
        #stds = np.squeeze(np.std(tempdata, axis=0))
        #ax4.errorbar(pos, np.squeeze(np.mean(tempdata, axis=0)), stds, lw=3)
       
        for k,td in enumerate(tempdata.transpose()):
            ax4.boxplot(td[~np.isnan(td)] , positions=[pos[k]], showbox=True, notch=True, showcaps=False, whis=0, showfliers=False, 
            boxprops={'color':cols[i]}, whiskerprops={'color':cols[i]}, flierprops={'color':cols[i]}, 
            bootstrap=5000)  #, , medianprops={'color':cols[i]}
    ax4.set_xlim(0, nmethods*NUM_SAMPLES)
    ax4.set_xticks(np.arange(0,NUM_SAMPLES*nmethods,sample_bars*nmethods)+2)
    ax4.set_xticklabels(list(range(1,NUM_SAMPLES+1, sample_bars)))    
    ax4.grid(True, which='major', axis='y')
    ax4.legend(loc=0, prop={'size':10})
    ax4.set_xlabel('Number of samples')
    # ax4.set_title('Path cost error relative to {0} method'.format(labels[comparison]))
    ax4.set_ylabel('Mean path cost relative to {0} (\%)'.format(labels[comparison]))  
    
    fig5, ax5 = plt.subplots()
    fig5.set_size_inches(fig_size+1, fig_size)
    percent_win = np.zeros((nmethods,NUM_SAMPLES))
    for jj in comp_methods:
        for kk in range(NUM_SAMPLES):
            percent_win[jj,kk] = (true_path_cost[:,jj,kk] < true_path_cost[:,comparison,kk]).sum()
        ax5.plot(np.arange(NUM_SAMPLES)+1, percent_win[jj,:]/NUM_STATRUNS*100.0, color=cols[jj], label=labels[jj])
    ax5.set_xlim(0, NUM_SAMPLES+1)
    ax5.grid(True, which='major', axis='y')
    ax5.legend(loc=0, prop={'size':10})
    ax5.set_xlabel('Number of samples')
    ax5.set_ylabel('Lower path cost vs {0} (\%)'.format(labels[comparison]))
    
    return fig1, fig2, fig3, fig4, fig5
    
    

    #lowcost_rand = lowcost_method.transpose()[:,0]/NUM_STATRUNS*100;
    #lowcost_maxv = lowcost_method.transpose()[:,1]/NUM_STATRUNS*100;
    #lowcost_fm = lowcost_method.transpose()[:,2]/NUM_STATRUNS*100;
    #h_lowcost = [ax3.bar(np.arange(0.75, NUM_SAMPLES), lowcost_rand, 0.5, color=cols[0], label=labels[0])]
    #h_lowcost.append(ax3.bar(np.arange(0.75, NUM_SAMPLES), lowcost_maxv, 0.5, bottom=lowcost_rand, color=cols[1], label=labels[1]))
    #h_lowcost.append(ax3.bar(np.arange(0.75, NUM_SAMPLES), lowcost_fm, 0.5, bottom=(lowcost_rand+lowcost_maxv), color=cols[2], label=labels[2]))



    #fig4, ax4 = plt.subplots()
    #fig4.set_size_inches(fig_size+1, fig_size)
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
    #ax4.set_xlabel('Number of samples')
    ## ax4.set_title('Path cost error relative to {0} method'.format(labels[comparison]))
    #ax4.set_ylabel('Relative path cost error (\%)'.format(labels[comparison]))  