#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os, re, glob

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(OUT_DIR)

datasets_label = ["Stanford\nCars","Oxford\nPets","Oxford\nFlowers","Food101","FGVC\nAircraft","UCF101","DTD","EuroSAT"]
datasets_short = ["Stanford Cars","Oxford Pets","Oxford Flowers","Food101","FGVC Aircraft","UCF101","DTD","EuroSAT"]
dataset_dirs  = ["stanford_cars","oxford_pets","oxford_flowers","food101","fgvc_aircraft","ucf101","dtd","eurosat"]

dlmpt_base  = [74.56,95.38,97.28,89.96,36.53,83.07,79.32,84.90]
dlmpt_novel = [71.93,97.58,69.10,90.46,31.77,65.44,49.60,56.35]
dlmpt_hm    = [2*b*n/(b+n) for b,n in zip(dlmpt_base,dlmpt_novel)]
paper_base  = [77.43,95.67,98.13,90.43,37.22,82.63,79.05,86.51]
paper_novel = [66.55,96.59,67.52,87.44,27.22,64.96,45.49,59.79]
paper_hm    = [2*b*n/(b+n) for b,n in zip(paper_base,paper_novel)]
delta_novel = [dn-pn for dn,pn in zip(dlmpt_novel,paper_novel)]

def parse_progress_logs(dataset_dir):
    pattern = os.path.join(PROJ_ROOT,"output","dlmpt",dataset_dir,"seed*","lambda0.2","progress.log")
    files = sorted(glob.glob(pattern))
    if not files: return None
    epoch_data = {}
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                m = re.search(r"epoch=(\d+)/\d+\s+batch=\d+/\d+\s+L_base=([\d.]+)\s+L_meta=([\d.]+)\s+L_total=([\d.]+)\s+acc_base=([\d.]+)%\s+acc_meta=([\d.]+)%",line)
                if not m: continue
                ep = int(m.group(1))
                if ep not in epoch_data: epoch_data[ep]={"L_base":[],"L_meta":[],"L_total":[],"acc_base":[],"acc_meta":[]}
                epoch_data[ep]["L_base"].append(float(m.group(2)))
                epoch_data[ep]["L_meta"].append(float(m.group(3)))
                epoch_data[ep]["L_total"].append(float(m.group(4)))
                epoch_data[ep]["acc_base"].append(float(m.group(5)))
                epoch_data[ep]["acc_meta"].append(float(m.group(6)))
    if not epoch_data: return None
    epochs = sorted(epoch_data.keys())
    return {"epoch":epochs,**{k:[np.mean(epoch_data[ep][k]) for ep in epochs] for k in ["L_base","L_meta","L_total","acc_base","acc_meta"]}}

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":14,"axes.labelsize":12,"xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":9,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":0.3,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.1})
CBLUE="#0072B2"; CORANGE="#E69F00"; CGREEN="#009E73"; CRED="#D55E00"; CPURPLE="#CC79A7"

def fig1():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.5),gridspec_kw={"width_ratios":[2.2,1]})
    x=np.arange(8); w=0.35
    ax1.bar(x-w/2,paper_novel,w,color=CBLUE,alpha=0.85,label="Paper CoOp+ATP (100ep)",edgecolor="white",linewidth=0.5)
    ax1.bar(x+w/2,dlmpt_novel,w,color=CORANGE,alpha=0.85,label="DL-MPT(CoOp+ATP) (25ep)",edgecolor="white",linewidth=0.5)
    for i,(dv,pv) in enumerate(zip(dlmpt_novel,paper_novel)):
        c=CGREEN if delta_novel[i]>0 else CRED
        ax1.annotate(f"{delta_novel[i]:+.2f}%",(x[i]+w/2,max(dv,pv)+2.0),ha="center",va="bottom",fontsize=8.5,fontweight="bold",color=c)
    ax1.set_xticks(x); ax1.set_xticklabels(datasets_label)
    ax1.set_ylabel("Novel Accuracy (%)"); ax1.set_title("Novel-Class Accuracy: DL-MPT vs Paper CoOp+ATP")
    ax1.legend(loc="lower left",frameon=True,fancybox=True); ax1.set_ylim(20,105)
    idx=np.argsort(delta_novel)
    colors=[CGREEN if delta_novel[i]>0 else CRED for i in idx]
    ax2.barh(range(8),[delta_novel[i] for i in idx],color=colors,edgecolor="white",linewidth=0.5)
    ax2.set_yticks(range(8)); ax2.set_yticklabels([datasets_short[i] for i in idx])
    ax2.axvline(0,color="black",linewidth=0.8); ax2.set_xlabel("Δ Novel (%)"); ax2.set_title("Novel Improvement (Sorted)")
    for i,dv in enumerate([delta_novel[j] for j in idx]):
        ax2.text(dv+(0.25 if dv>=0 else -0.25),i,f"{dv:+.2f}",va="center",fontsize=9,fontweight="bold",color=colors[i])
    fig.suptitle("Figure 1: DL-MPT(CoOp+ATP) Novel-Class Performance",fontsize=15,fontweight="bold",y=1.02)
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"fig1_delta_novel.png"),facecolor="white"); plt.close(fig)
    print("[OK] fig1_delta_novel.png")

def fig2():
    fig,axes=plt.subplots(2,4,figsize=(18,9)); axes=axes.flatten()
    for i in range(8):
        ax=axes[i]; b1,n1=dlmpt_base[i],dlmpt_novel[i]; b2,n2=paper_base[i],paper_novel[i]
        dv=[b1,n1,2*b1*n1/(b1+n1)]; pv=[b2,n2,2*b2*n2/(b2+n2)]
        x=np.arange(3); w=0.32
        ax.bar(x-w/2,pv,w,color=CBLUE,alpha=0.85,label="Paper CoOp+ATP",edgecolor="white",linewidth=0.5)
        ax.bar(x+w/2,dv,w,color=CORANGE,alpha=0.85,label="DL-MPT",edgecolor="white",linewidth=0.5)
        for j in range(3):
            d=dv[j]-pv[j]
            if abs(d)>0.1:
                c=CGREEN if d>0 else CRED; va="bottom" if d>0 else "top"; yo=1.8 if d>0 else -1.8
                ax.annotate(f"{d:+.2f}",(x[j]+w/2,max(dv[j],pv[j])+yo),ha="center",va=va,fontsize=7,fontweight="bold",color=c)
        ax.set_xticks(x); ax.set_xticklabels(["Base","Novel","HM"]); ax.set_title(datasets_short[i],fontsize=11,fontweight="bold")
        if i==0: ax.legend(fontsize=7,loc="upper right")
    fig.suptitle("Figure 2: Per-Dataset Base / Novel / HM Comparison",fontsize=15,fontweight="bold",y=1.01)
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"fig2_per_dataset.png"),facecolor="white"); plt.close(fig)
    print("[OK] fig2_per_dataset.png")

def fig3():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.5))
    y_pos=range(8)
    for i in y_pos: ax1.plot([paper_novel[i],dlmpt_novel[i]],[i,i],color="gray",linewidth=1.2,alpha=0.6,zorder=1)
    ax1.scatter(paper_novel,y_pos,s=80,color=CBLUE,alpha=0.8,zorder=2,label="Paper CoOp+ATP")
    ax1.scatter(dlmpt_novel,y_pos,s=80,color=CORANGE,alpha=0.8,zorder=3,marker="D",label="DL-MPT(CoOp+ATP)")
    for i in y_pos:
        d=delta_novel[i]; c=CGREEN if d>0 else CRED
        ax1.text(max(paper_novel[i],dlmpt_novel[i])+1.5,i,f"{d:+.1f}",va="center",fontsize=9,fontweight="bold",color=c)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(datasets_short)
    ax1.set_xlabel("Novel Accuracy (%)"); ax1.set_title("Novel Accuracy Change (Paired)"); ax1.legend(loc="lower right",fontsize=9)
    ax1.set_xlim(min(min(paper_novel),min(dlmpt_novel))-5,max(max(paper_novel),max(dlmpt_novel))+8)

    ax2.scatter(paper_hm,dlmpt_hm,s=70,color=CPURPLE,alpha=0.8,edgecolors="white",linewidth=0.6)
    for i,ds in enumerate(datasets_short):
        ax2.annotate(ds,(paper_hm[i],dlmpt_hm[i]),textcoords="offset points",xytext=(5,5),fontsize=7.5,alpha=0.8)
    lims=[45,100]
    ax2.plot(lims,lims,"--",color="gray",alpha=0.5,label="y = x")
    ax2.set_xlabel("Paper CoOp+ATP HM (%)"); ax2.set_ylabel("DL-MPT(CoOp+ATP) HM (%)")
    ax2.set_title("Harmonic Mean Comparison"); ax2.legend(fontsize=8)
    ax2.set_xlim(lims); ax2.set_ylim(lims); ax2.set_aspect("equal")
    fig.suptitle("Figure 3: Summary View — Improvement Consistency",fontsize=15,fontweight="bold",y=1.02)
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"fig3_summary.png"),facecolor="white"); plt.close(fig)
    print("[OK] fig3_summary.png")

def fig4():
    fig,ax=plt.subplots(figsize=(10,4))
    epochs=np.arange(1,26)
    lambdas=[0.0 if ep<=5 else (0.5 if ep>=21 else 0.2) for ep in epochs]
    ax.step(epochs,lambdas,where="post",color=CPURPLE,linewidth=2.5,label="\u03bb (meta weight)")
    ax.fill_between(epochs,0,lambdas,step="post",alpha=0.15,color=CPURPLE)
    ax.axvspan(0.5,5.5,alpha=0.08,color="blue"); ax.axvspan(5.5,20.5,alpha=0.08,color="green"); ax.axvspan(20.5,25.5,alpha=0.08,color="red")
    ax.text(3,0.06,"Warmup\n(ep 1-5)",ha="center",fontsize=9,color="blue",fontweight="bold")
    ax.text(13,0.26,"Joint (ep 6-20)",ha="center",fontsize=9,color="green",fontweight="bold")
    ax.text(23,0.56,"Refine\n(ep 21-25)",ha="center",fontsize=9,color="red",fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("\u03bb (meta loss weight)"); ax.set_title("Figure 4: Three-Stage \u03bb Scheduling")
    ax.set_xlim(0.5,25.5); ax.set_ylim(0,0.65); ax.legend(loc="upper left",fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"fig4_lambda_schedule.png"),facecolor="white"); plt.close(fig)
    print("[OK] fig4_lambda_schedule.png")

def fig5():
    all_data={}
    for ds_name,ds_dir in zip(datasets_short,dataset_dirs):
        data=parse_progress_logs(ds_dir)
        if data: all_data[ds_name]=data
    if not all_data: print("[WARN] No training logs for fig5!"); return
    fig,axes=plt.subplots(4,2,figsize=(18,18)); axes_flat=axes.flatten()
    colors_8=plt.cm.tab10(np.linspace(0,1,len(all_data)))
    panels=[("L_base","(a) L_base — Base Classification Loss","upper right"),
            ("L_meta","(b) L_meta — Episodic Meta Loss","upper right"),
            ("L_total","(c) L_total = L_base + \u03bb\u00b7L_meta","upper right"),
            ("acc_base","(d) Base Accuracy (training)","lower right"),
            ("acc_meta","(e) Meta Episode Accuracy","lower right"),
            (None,"(f) Meta Loss Ratio L_meta/L_total","upper right")]
    for idx,(key,title,legend_loc) in enumerate(panels):
        ax=axes_flat[idx]
        for j,(ds_name,data) in enumerate(all_data.items()):
            if key is None:
                vals=[m/t if t>0 else 0 for m,t in zip(data["L_meta"],data["L_total"])]
            else:
                vals=data[key]
            ax.plot(data["epoch"],vals,color=colors_8[j],linewidth=1.5,label=ds_name,alpha=0.85)
        ax.set_xlabel("Epoch"); ax.set_ylabel(key if key else "L_meta/L_total")
        ax.set_title(title); ax.legend(fontsize=6.5,ncol=2,loc=legend_loc); ax.set_xlim(1,25)

    ax=axes_flat[6]
    all_lb=np.array([data["L_base"] for data in all_data.values()])
    epochs=list(all_data.values())[0]["epoch"]
    ax.plot(epochs,all_lb.mean(axis=0),color="black",linewidth=2,label="mean L_base")
    ax.fill_between(epochs,all_lb.mean(axis=0)-all_lb.std(axis=0),all_lb.mean(axis=0)+all_lb.std(axis=0),alpha=0.2,color="black")
    ax.set_xlabel("Epoch"); ax.set_ylabel("L_base"); ax.set_title("(g) Aggregate L_base (mean \u00b1 std)"); ax.legend(fontsize=8); ax.set_xlim(1,25)

    ax=axes_flat[7]
    all_am=np.array([data["acc_meta"] for data in all_data.values()])
    ax.plot(epochs,all_am.mean(axis=0),color="black",linewidth=2,label="mean acc_meta")
    ax.fill_between(epochs,all_am.mean(axis=0)-all_am.std(axis=0),all_am.mean(axis=0)+all_am.std(axis=0),alpha=0.2,color="black")
    ax.set_xlabel("Epoch"); ax.set_ylabel("acc_meta (%)"); ax.set_title("(h) Aggregate Meta Accuracy (mean \u00b1 std)"); ax.legend(fontsize=8); ax.set_xlim(1,25)

    fig.suptitle("Figure 5: DL-MPT Training Dynamics (8 datasets)",fontsize=16,fontweight="bold",y=1.005)
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"fig5_training_dynamics.png"),facecolor="white"); plt.close(fig)
    print("[OK] fig5_training_dynamics.png")

def fig6():
    all_data={}
    for ds_name,ds_dir in zip(datasets_short,dataset_dirs):
        data=parse_progress_logs(ds_dir)
        if data: all_data[ds_name]=data
    highlight=["Stanford Cars","Food101","DTD"]
    available=[d for d in highlight if d in all_data]
    if not available: print("[WARN] No data for fig6!"); return
    fig,axes=plt.subplots(1,len(available),figsize=(6*len(available),4.5))
    if len(available)==1: axes=[axes]
    for ax,ds_name in zip(axes,available):
        data=all_data[ds_name]; epochs=data["epoch"]
        ax.plot(epochs,data["L_base"],color=CBLUE,linewidth=2,label="L_base")
        ax.plot(epochs,data["L_meta"],color=CORANGE,linewidth=2,label="L_meta")
        ax.plot(epochs,data["L_total"],color="black",linewidth=1.5,linestyle="--",label="L_total")
        for ep in [5,20]:
            if ep in epochs:
                idx=epochs.index(ep); ax.axvline(ep,color="gray",linestyle=":",alpha=0.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(ds_name,fontweight="bold"); ax.legend(fontsize=8); ax.set_xlim(1,25)
    fig.suptitle("Figure 6: Per-Dataset Loss Decomposition",fontsize=15,fontweight="bold",y=1.03)
    plt.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"fig6_loss_detail.png"),facecolor="white"); plt.close(fig)
    print("[OK] fig6_loss_detail.png")

if __name__=="__main__":
    print("Generating DL-MPT(CoOp+ATP) figures...")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print(f"\nDone! All figures saved to {OUT_DIR}/")
