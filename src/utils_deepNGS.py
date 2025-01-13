import os
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from pandas.api.types import is_numeric_dtype
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
AAS = list("RKHDESTNQAVILMFYWGPC-*")
AA2I = {aa: i for i, aa in enumerate(AAS)}


def ResiConvert(line,array,trueletter):
    for item in array:
        line = line.replace(item,trueletter)
    return line



# Generalise AAs based on their groups
def SecondSetGroupConversion(df):
    # Remember to use sequence!
    HGroup = ['A','L','V','I','M']
    PGroup = ['S','T','C','N','Q']
    AGroup = ['F','Y','W']
    BGroup = ['H','K','R']
    NGroup = ['D','E']
    SGroup = ['G','P']
    OneGroup = ['1']
    TwoGroup = ['2']
    ThreeGroup = ['3']
    FourGroup = ['4']
    FiveGroup = ['5']
    SixGroup = ['6']
    RowCount = df['Seq'].count()

    for i in range(0,RowCount):

        TruthI = i
        OrigSeq = df.at[i,'Seq']
        Seq = ResiConvert(OrigSeq,HGroup,'1')
        Seq = ResiConvert(Seq,PGroup,'2')
        Seq = ResiConvert(Seq,AGroup,'3')
        Seq = ResiConvert(Seq,BGroup,'4')
        Seq = ResiConvert(Seq,NGroup,'5')
        Seq = ResiConvert(Seq,SGroup,'6')
        Seq = ResiConvert(Seq,OneGroup,'H')
        Seq = ResiConvert(Seq,TwoGroup,'P')
        Seq = ResiConvert(Seq,ThreeGroup,'A')
        Seq = ResiConvert(Seq,FourGroup,'B')
        Seq = ResiConvert(Seq,FiveGroup,'N')
        Seq = ResiConvert(Seq,SixGroup,'S')
        df.at[TruthI,'Seq'] = Seq


class Utils:
    def __init__(self):
        # Get the path of execution
        self.current_path = os.getcwd()
        # Set the current location
        os.chdir(self.current_path)

    def set_path_and_arguments(self,fileName):
        """
        Set up file paths and command line arguments.
        """
        self.fileName = fileName
        self.figdir = f"{self.current_path}/outputs/deepngs/figs/{self.fileName}/"
        self.storedir = f"{self.current_path}/outputs/deepngs/processed_deepNGS_files/"
        os.makedirs(self.figdir, exist_ok=True)
        os.makedirs(self.storedir, exist_ok=True)


    def preprocess_data(self, file_path,project):
        """
        read df and adjust columns
        """
        self.df = pd.read_csv(f"{self.current_path}/outputs/deepngs/{file_path}.csv.gz")

        if 'picked_clone' not in self.df.columns:
            self.df['picked_clone']=''
        if 'affinity_metric' not in self.df.columns:
            self.df['affinity_metric']=-1
        self.df=self.df.reset_index(drop=True)

        # clc charge & hydrophobicity
        self.df['cdr3_']=self.df['CDR3'].str.replace('-','')
        self.df['pr'] = self.df['cdr3_'].apply(ProteinAnalysis)
        self.df['charge_pH7_cdr3']=[pr.charge_at_pH(7.4) for pr in self.df['pr'].values]
        self.df['hydrophobicity_cdr3']=[self.catch_gravy(pr) for pr in self.df['pr'].values]
        self.df['seq_']=self.df['AA'].str.replace('-','')
        self.df['pr']=self.df['seq_'].apply(ProteinAnalysis)
        self.df['charge_pH7_fullSeq']=[pr.charge_at_pH(7.4) for pr in self.df['pr'].values]
        self.df['hydrophobicity_fullSeq']=[self.catch_gravy(pr) for pr in self.df['pr'].values]
        self.df.drop(['pr','seq_','cdr3_'],axis=1,inplace=True)

        # find generalised cdr3
        SequenceDF = self.df['CDR3'].reset_index()
        SequenceDF.columns=['Title','Seq']
        SecondSetGroupConversion(SequenceDF)
        self.df['cdr3_generalised']=SequenceDF['Seq'].values
        self.df.to_csv(f'{self.storedir}/{self.fileName}_processed_data.tsv.gz',sep='\t',compression='gzip')

    def catch_gravy(self, pr):
        """
        Calculate GRAVY (Grand Average of Hydropathy) score for a protein sequence.
        Parameters:
        - pr (ProteinAnalysis): ProteinAnalysis object for a protein sequence.
        Returns:
        - float: GRAVY score.
        """
        try:
            return pr.gravy()
        except:
            return 0

    def plot_with_binders(self, df, colorby, colorby_name, fname, colorby_type="continuous", clim=None, cmap="Spectral_r"):
        """
        Plot data points with binders highlighted.

        Parameters:
        - df (DataFrame): DataFrame containing the data.
        - colorby (str): Column name to use for coloring data points.
        - colorby_name (str): Name of the color-by column.
        - fname (str): File name to save the plot.
        - colorby_type (str): Type of the color-by column (continuous or categorical).
        - clim (tuple): Color limit for continuous color-by column.
        - cmap (str): Colormap for continuous color-by column.
        """
        plt.close()
        fig, ax = plt.subplots(figsize=(10, 10))
        pt_size = 3  # Fixed point size

        # Calculate the axis limits early to apply later
        x_min, x_max = df['e1'].min(), df['e1'].max()
        y_min, y_max = df['e2'].min(), df['e2'].max()
        x_pad = (x_max - x_min) * 0.05  # Adding 5% padding
        y_pad = (y_max - y_min) * 0.05

        if colorby_type == "continuous":
            df.sort_values(colorby, inplace=True)
            norm = Normalize(vmin=clim[0], vmax=clim[1]) if clim else Normalize(vmin=df[colorby].min(), vmax=df[colorby].max())
            cmap_instance = sns.color_palette(cmap, as_cmap=True)
            scatter = ax.scatter(df["e1"], df["e2"], c=df[colorby], cmap=cmap_instance,  norm=norm, s=pt_size, edgecolor=None, linewidth=0.5)#
            # Create colorbar and adjust its size relative to the main plot
            cbar = plt.colorbar(scatter, ax=ax, label=colorby_name, fraction=0.046, pad=0.04)
            top_categories=None
        else:
            # Handle categorical data
            category_counts = df[colorby].value_counts()
            rare_categories = ['rare']
            # Define color palette
            palette = {cat: 'lightgrey' if cat in rare_categories else None for cat in df[colorby].unique()}
            remaining_palette = sns.color_palette(cmap, n_colors=df[colorby].nunique() )
            i = 0
            for cat in category_counts.index:
                if palette[cat] is None:
                    palette[cat] = remaining_palette[i]
                    i += 1

            df['colorby_legend'] = df[colorby].apply(lambda x: 'rare' if x in rare_categories else x)
            sns.scatterplot(data=df, x="e1", y="e2", hue='colorby_legend', palette=palette, ax=ax, s=pt_size, edgecolor=None)

        df_1 = df[(~df["picked_clone"].isna())&(df["picked_clone"]!="")]
        if df_1.shape[0]>0:
            sns.scatterplot(data=df_1,x="e1", y="e2" ,
                    s=80., edgecolor='black', facecolor="none", style='picked_clone',linewidths=1,ax=ax,alpha=0.7)

        # Additional plotting logic...
        ax.set_title(f"N={len(df)}")
        adjust_pos = -0.3 if colorby_type == "continuous" else (-0.3 if df[colorby].nunique() > 12 else -0.3)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, adjust_pos), ncol=4, fancybox=True, framealpha=0.5, prop={'size': 8})
        ax.set(ylim=(y_min - y_pad, y_max + y_pad),xlim=(x_min - x_pad, x_max + x_pad))
        ax.set_title(f"Plot of {colorby_name} #{df.shape[0]}")
        ax.axis('scaled')
        ax.legend(loc='center left', bbox_to_anchor=(1.22, 0.5))
        plt.tight_layout()
        plt.savefig(fname, format='png', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_analysis_figures(self):
        """
        Plot analysis figures based on the input DataFrame.

        Parameters:
        df_s (DataFrame): DataFrame containing the data for analysis.
        """
        numerical_global=['charge_pH7_cdr3','charge_pH7_fullSeq','hydrophobicity_cdr3','hydrophobicity_fullSeq','log10_num_neighbors','length']
        catehgorical_global=['HCDR1','HCDR2','HCDR3','LCDR1','LCDR2','LCDR3','method','affinity_FACS_label','cdr3_generalised','animal','len-len_cdr','Target','v_gene','j_gene','source','label']
        df_s=self.df
        numerical   =[x for x in numerical_global if x in df_s.columns]
        catehgorical=[x for x in catehgorical_global if x in df_s.columns]

        for prop in  numerical:
            if df_s[prop].drop_duplicates().shape[0]>1:
                fname = f"{self.figdir}/{prop}.png"
                self.plot_with_binders(df_s, colorby=f"{prop}", colorby_name=f"{prop}", fname=fname)

        for prop in  catehgorical:
            fname = f"{self.figdir}/{prop}.png"
            df_s2=df_s.copy()
            size_=df_s[prop].unique().shape[0]
            if is_numeric_dtype(df_s[prop]):
                df_s2[prop].fillna(-1,inplace=True)
                df_s2[prop]=df_s2[prop].astype(str)
            if size_>20:
                df_s2[f'{prop}_']=df_s2[f'{prop}']
                top_20=list(df_s2[f'{prop}_'].value_counts()[0:20].index.values)
                df_n= df_s2[~df_s2[f'{prop}_'].isin(top_20)]
                df_n[f'{prop}_']='rare'
                df_p = df_s2[df_s2[f'{prop}_'].isin(top_20)]
                df_s2 =pd.concat([df_n, df_p])
                df_s2.fillna('',inplace=True)
                self.plot_with_binders(df_s2, colorby=f'{prop}_', colorby_name=f'{prop}', fname=fname, colorby_type="categorical")
            else:
                df_s2=df_s2[~df_s2[prop].isna()]
                df_s2.fillna('',inplace=True)
                print(df_s[prop].value_counts())
                self.plot_with_binders(df_s2, colorby=f'{prop}', colorby_name=f'{prop}', fname=fname, colorby_type="categorical")
        return self.df