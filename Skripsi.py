import streamlit as st
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import seaborn as sns

# import for clustering
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from yellowbrick.cluster import SilhouetteVisualizer
import copy
from PIL import Image , ImageOps
from streamlit_option_menu import option_menu

import statistics

# To Improve speed and cache data
@st.cache(persist=True)
def dataSet(retail):
	retail = pd.read_csv(retail, encoding= 'unicode_escape')
	return retail 

# Init Centroid base on cutTree Hierarchical Clustering for K Means Clustering + Euclidean Distance
def Initiate_Centroid(rfm_df_scaled,k):
    Centroid = []
    for i in range(k):
        initC = rfm_df_scaled[rfm_df_scaled['Cluster Hierarchical'] == i]

        C_Amount = initC.iloc[:,0]
        C_Freq = initC.iloc[:,1]
        C_Recency = initC.iloc[:,2]

        CAmount_ = C_Amount.to_numpy()
        CFreq_ = C_Freq.to_numpy()
        CRecency_ = C_Recency.to_numpy()

        CAmount_Mean = statistics.fmean(CAmount_)
        CFreq_Mean = statistics.fmean(CFreq_)
        CRecency_Mean = statistics.fmean(CRecency_)

        C = [CAmount_Mean, CFreq_Mean, CRecency_Mean]
        Centroid.append(C)

    Centroid_df = pd.DataFrame(Centroid)
    return Centroid_df

# Using Euclidean Distance
def Calc_Dist(rfm_df_scaled,m,f,r,Centroid_df,k):
      for i in range(k):
            rfm_df_scaled[str(i)] = np.sqrt((m - Centroid_df.iloc[i,0]) ** 2 + (f - Centroid_df.iloc[i,1]) ** 2 + (r - Centroid_df.iloc[i,2]) ** 2) 
      return rfm_df_scaled

def df_membership(rfm_df_scaled2,k):
    clt = k
    index = 1
    rfm_df_scaled2['Index Cluster'] = (rfm_df_scaled2.loc[:, ['{}'.format(index) for index in range(clt)]].idxmin(axis=1)).astype('int') 
    return rfm_df_scaled2

def rearrange_centroid(rfm_df_scaled2,k):
  Centroid_df_new = []
  for i in range(k):
    initC_new = rfm_df_scaled2[rfm_df_scaled2['Index Cluster'] == i]

    C_Amount_new = initC_new.iloc[:,0]
    C_Freq_new = initC_new.iloc[:,1]
    C_Recency_new = initC_new.iloc[:,2]

    CAmount_new_ = C_Amount_new.to_numpy()
    CFreq_new = C_Freq_new.to_numpy()
    CRecency_new = C_Recency_new.to_numpy()

    CAmount_Mean_new = statistics.fmean(CAmount_new_)
    CFreq_Mean_new = statistics.fmean(CFreq_new)
    CRecency_Mean_new = statistics.fmean(CRecency_new)

    C_new = [CAmount_Mean_new, CFreq_Mean_new, CRecency_Mean_new]
    Centroid_df_new.append(C_new)

  Centroid_df2 = pd.DataFrame(Centroid_df_new)
  return Centroid_df2


# Fungsi Clusterisasi K Means 
def clusterisasi(rfm_df_scaled,m,f,r,k,Centroid_df):
  iterasi = 1
  centroid_df = Initiate_Centroid(rfm_df_scaled, k)
  rfm_df_scaled2 = copy.deepcopy(rfm_df_scaled)
  rfm_df_scaled2 = Calc_Dist(rfm_df_scaled2,m,f,r,Centroid_df,k)
  rfm_df_scaled2 = df_membership(rfm_df_scaled2,k)
  centroid_df = rearrange_centroid(rfm_df_scaled2,k)
  while (True):
    iterasi = iterasi + 1
    oldcentroid = copy.deepcopy(centroid_df)
    rfm_df_scaled2 = Calc_Dist(rfm_df_scaled2,m,f,r,Centroid_df,k)
    rfm_df_scaled2 = df_membership(rfm_df_scaled2,k)
    centroid_df = rearrange_centroid(rfm_df_scaled2,k)
    if (oldcentroid.equals(centroid_df)):
      break
  return (rfm_df_scaled2,centroid_df, iterasi)

# Fungsi Output RFM (std dev)
def output_stdDev(rfm_df_scaled2, k):
  stdDev_list = []
  for i in range(k):
      init_StdDev = rfm_df_scaled2[rfm_df_scaled2['Index Cluster'] == i]

      Amount = init_StdDev.iloc[:,0]
      Freq = init_StdDev.iloc[:,1]
      Recency = init_StdDev.iloc[:,2]

      Amount_np = Amount.to_numpy()
      Freq_np = Freq.to_numpy()
      Recency_np = Recency.to_numpy()

      Amount_stdDev = np.std(Amount_np)
      CFreq_stdDev = np.std(Freq_np)
      CRecency_stdDev = np.std(Recency_np)

      stdDev = [Amount_stdDev, CFreq_stdDev, CRecency_stdDev]
      stdDev_list.append(stdDev)

  stdDev_df = pd.DataFrame(stdDev_list)
  stdDev_df.columns = ['M', 'F', 'R']
  return stdDev_df

# Region Function for Session State
if "Clt_Res" not in st.session_state:
  st.session_state.Clt_Res = False

def Clt_Res_Callback():
  st.session_state.Clt_Res = True

# endregion

@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def main():
    # Navbar
    menu = option_menu(
      menu_title=None,
      options=['Home', 'Guide', 'About'],
      menu_icon='cast',
      default_index=0,
      orientation='horizontal',
    )
    # Title App
    # st.title("""Customer Segmentation Based On RFM""")
    st.markdown("<h1 style='text-align: center;'>Customer Segmentation Based On RFM</h1>", unsafe_allow_html=True)
    st.markdown("""---""")

    if menu == 'Home':
      # Upload Data
      st.sidebar.subheader('Upload Dataset')

      # Setup file upload 
      retail = st.sidebar.file_uploader(label='Support CSV', type=['csv'], accept_multiple_files= False)

      if retail is not None:
          retail = dataSet(retail)

          # slider untuk klaster
          k = st.sidebar.slider(min_value = 2, max_value = 10, value = 2, label = 'Number of Clusters')
          # Select box ntuk method hierarchical clustering
          hierarchical_method = st.sidebar.selectbox('Hierarchical Method', ('single', 'complete', 'average', 'ward'))

          if (st.sidebar.button(label = 'Hierarchical K-Means Clustering', help = 'for clustering', on_click=Clt_Res_Callback) or st.session_state.Clt_Res):
              # preprocessing data & drop data null
              retail = retail.dropna()

              retail['CustomerID'] = retail['CustomerID'].astype(str)

              # Attribute : Monetary
              retail['Amount'] = retail['Quantity']*retail['UnitPrice']
              rfm_m = retail.groupby('CustomerID')['Amount'].sum()
              rfm_m = rfm_m.reset_index()

              # Attribute : Frequency
              rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count()
              rfm_f = rfm_f.reset_index()
              rfm_f.columns = ['CustomerID', 'Frequency']

              # Merging the two dfs
              rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')

              # New Attribute : Recency
              # Convert to datetime to proper datatype
              retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')
              max_date = max(retail['InvoiceDate'])
              retail['Diff'] = max_date - retail['InvoiceDate']
              rfm_p = retail.groupby('CustomerID')['Diff'].min()
              rfm_p = rfm_p.reset_index()
              rfm_p['Diff'] = rfm_p['Diff'].dt.days
              rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')

              rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

              # Outliers Analysis
              # outliers for Amount
              Q1 = rfm.Amount.quantile(0.25)
              Q3 = rfm.Amount.quantile(0.75)
              IQR = Q3 - Q1
              rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

              # outliers for Recency
              Q1 = rfm.Recency.quantile(0.25)
              Q3 = rfm.Recency.quantile(0.75)
              IQR = Q3 - Q1
              rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

              # outliers for Frequency
              Q1 = rfm.Frequency.quantile(0.25)
              Q3 = rfm.Frequency.quantile(0.75)
              IQR = Q3 - Q1
              rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]

              # Rescaling the attributes
              rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
              scaler = StandardScaler()

              # fit_transform
              rfm_df_scaled = scaler.fit_transform(rfm_df)

              rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
              rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

              #perform hierarchical clustering here
              # Single Linkage
              _hierarchicalClustering = linkage(rfm_df_scaled, method=hierarchical_method, metric='euclidean')
              cutTree = cut_tree(_hierarchicalClustering, n_clusters=k).reshape(-1,)

              # Dendogram
              st.header(hierarchical_method + ' Method')
              fig_Silh = plt.figure(figsize=(20, 10)) 
              dendrogram(_hierarchicalClustering)
              st.pyplot(fig_Silh) 


              # assign cluster label
              rfm_df_scaled['Cluster Hierarchical'] = cutTree

              # Iniate Centroid for HK-Means
              Centroid_df = Initiate_Centroid(rfm_df_scaled, k)

              rfm_df_scaled2 = Calc_Dist(rfm_df_scaled, rfm_df_scaled['Amount'], rfm_df_scaled['Frequency'], rfm_df_scaled['Recency'], Centroid_df, k)

              # for Index Cluster
              rfm_df_scaled2 = df_membership(rfm_df_scaled2, k)


              # Final HKMeans
              rfm_df_scaled2, centroid_df , iterasi= clusterisasi(rfm_df_scaled, rfm_df_scaled['Amount'], rfm_df_scaled['Frequency'], rfm_df_scaled['Recency'], k, Centroid_df)

              Cluster_labels = rfm_df_scaled2['Index Cluster'].to_numpy()

              rfm['Hk Means Cluster'] = Cluster_labels

              # Region Silhouette Analysis
              # Silhouette analysis
              st.subheader("Silhouette Analysis")
              silhouette_df = []
              silhouette = []
              range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
              range_k = range(2,11)

              for num_clusters in range_n_clusters:
                  
                  cluster_labels = Cluster_labels
                  
                  # silhouette score
                  silhouette_avg = silhouette_score(rfm_df_scaled2, cluster_labels)
                  silhouette.append(silhouette_avg)
                  listSilhouette_df = [num_clusters, silhouette_avg]
                  silhouette_df.append(listSilhouette_df)
                  # st.text("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
              
              silhouette_df = pd.DataFrame(silhouette_df)
              silhouette_df.columns = ['Clusters', 'Silhouette Score']
              silhouette_df.style.hide(axis='index')

              st.write(silhouette_df)


              # Vizualization Silhouette Analysis   
              fig_Silh = plt.figure(figsize=(20, 10)) 
              plt.plot(range_k, silhouette, 'bx-')
              plt.xlabel('Values of K')
              plt.ylabel('silhouette_avg')
              plt.title('silhouette')
              st.pyplot(fig_Silh) 

              # Region show plot
              st.header('Monetary')
              fig_amt = plt.figure(figsize=(20, 10))
              sns.boxplot(x='Hk Means Cluster', y='Amount', data=rfm).set_title('Amount Plot')
              st.pyplot(fig_amt) 

              st.header('Frequency')
              fig_freq = plt.figure(figsize=(20, 10))
              sns.boxplot(x='Hk Means Cluster', y='Frequency', data=rfm).set_title('Frequency Plot')
              st.pyplot(fig_freq) 

              st.header('Recency')
              fig_rec = plt.figure(figsize=(20, 10))
              sns.boxplot(x='Hk Means Cluster', y='Recency', data=rfm).set_title('Recency Plot')
              st.pyplot(fig_rec) 

              # Output RFM
              st.header('RFM Output')
              rfm_img = Image.open('img/app_img/rfm.png')
              st.image(rfm_img, caption='RFM Mapper')

              stdDev_df = output_stdDev(rfm_df_scaled2, k)

              # Untuk Monetary avg
              m_ = stdDev_df.iloc[:, 0]
              m_arr = m_.to_numpy()
              avg_m = np.average(m_arr)
              # Untuk Freq avg
              f_ = stdDev_df.iloc[:, 1]
              f_arr = f_.to_numpy()
              avg_f = np.average(f_arr)
              # Untuk Recency avg
              r_ = stdDev_df.iloc[:, 2]
              r_arr = r_.to_numpy()
              avg_r = np.average(r_arr)

              # for monetary output
              col1,col2,col3 = st.columns(3)

              with col1:
                # for Monetary output
                for num_clusters in range(k):
                  if stdDev_df.iloc[num_clusters, 0] > avg_m:
                      m_output = 'Monetary at cluster {0} is M(up)'.format(num_clusters)
                  else:
                      m_output = 'Monetary at cluster {0} is M(down)'.format(num_clusters)
                  st.write(m_output)
              with col2:
                # for Frequency output
                for num_clusters in range(k):
                  if stdDev_df.iloc[num_clusters, 1] > avg_f:
                      f_output = 'Frequency at cluster {0} is F(up)'.format(num_clusters)
                  else:
                      f_output = 'Frequency at cluster {0} is F(down)'.format(num_clusters)
                  st.write(f_output)
              # with col3:
              #   # for Recency output
              #   for num_clusters in range(k):
              #     if stdDev_df.iloc[num_clusters, 2] > avg_r:
              #         r_output = 'Recency at cluster {0} is R(up)'.format(num_clusters)
              #     else:
              #         r_output = 'Recency at cluster {0} is R(down)'.format(num_clusters)
              #     st.write(r_output)


              # Region Sidebar continue
              if st.sidebar.checkbox(label = 'Show Data'):
                  # Show Data RFM H-K Means
                  st.header('Data RFM')
                  st.write(rfm)
                  # Download output data as csv
                  csv_RFM = convert_df(rfm)
                  st.sidebar.download_button(label = 'Save Data', data=csv_RFM, file_name='RFM_Hkmeans.csv', mime='text/csv')
              # endregion
    
    if menu == 'Guide':
      st.markdown("<h2 style='text-align: center;'>User Manual Guide</h2>", unsafe_allow_html=True)

      # Special notes
      st.subheader('Attention')
      st.markdown('<p style="text-align: justify;">Attention, for the time being this application can only be used with the following dataset : </p>', unsafe_allow_html=True)
      st.markdown("""<a href="https://storage.googleapis.com/kaggle-data-sets/397875/764509/compressed/OnlineRetail.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221113%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221113T142700Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=829fdd91d6b75f337b150604b0a8beb9c70383c4e7327be85eb597d66ef24af4635233daff2ef05f96221bc270aeea98139a510f80b12e093814bf82425027891711b4e00106a1289a82aadd955856447410eb71ef098cc7d1f26b98c22ad1f06cbb3b1259132e90373272bc6fd8823f020da83fdf5768455cab4388f0c28dfc140de11f2e72abf72aade371cb1f65cf98467509fff716411aee64bce07ed4ddee58a74019bff40b3e55748e73dcf9e0bc1a0c377e63528d0b79225fa290d3185db9411b9be92d669fd4085225ac1769137c491b7fdca562e43786e0e4128bfd8117a46137f4fa28b49da57e2d147bb4680a40a331fd108d82f9dfe44ba469b6">OnlineRetail.csv</a>""", unsafe_allow_html=True)

      # Home
      st.subheader('1. Home')
      home_img = ImageOps.expand(Image.open('img/user_manual/home.png'), border= 5, fill='white')
      st.image(home_img, caption='Home')
      st.markdown('<p style="text-align: justify;">home is a page where users can perform customer segmentation computations using the hierarchical K - Means clustering method</p>', unsafe_allow_html=True)

      # Upload
      st.subheader('1.1. Upload Dataset')
      uploadData_img = ImageOps.expand(Image.open('img/user_manual/uploadDataset.png'), border= 5, fill='white')
      st.image(uploadData_img, caption='Upload Dataset')
      st.markdown('<p style="text-align: justify;">Users can upload data which will later be used as data for computing customer segmentation calculations, how to upload data is to click "Browse files", and select the data.</p> <p style="color:#FF0000";>Notes: this application only supports data of type csv only!!</p>', unsafe_allow_html=True)
      uploadDataAfter_img = ImageOps.expand(Image.open('img/user_manual/uploadDataset_after.png'), border= 5, fill='white')
      st.image(uploadDataAfter_img, caption='After Upload Dataset')

      # Number of Clusters
      st.subheader('1.2. Number of Clusters')
      slider_numClt_img = ImageOps.expand(Image.open('img/user_manual/slider_numClt.png'), border= 5, fill='white')
      st.image(slider_numClt_img, caption='Slider Number of Clusters')
      st.markdown('<p style="text-align: justify;">The Number of Clusters slider functions to determine the desired number of clusters before carrying out the clustering process. The way to determine the desired cluster is to move the slider point with a value range of 2 to 10</p>', unsafe_allow_html=True)

      # Number of Clusters
      st.subheader('1.3. Hierarchical Method')
      hierarchicalMethod_img = ImageOps.expand(Image.open('img/user_manual/hierarchicalMethod.png'), border= 5, fill='white')
      st.image(hierarchicalMethod_img, caption='Hierarchical Method')
      st.markdown('<p style="text-align: justify;">There are 4 choices in determining the hierarchical clustering method, namely single(This is also known as the Nearest Point Algorithm), complete(Farthest Point Algorithm or Voor Hees Algorithm), average(This is also called the UPGMA algorithm), and ward(uses the Ward variance minimization algorithm, this is also known as the incremental algorithm)</p>', unsafe_allow_html=True)

      # Hierarchical K-Means Clustering
      st.subheader('1.4. Hierarchical K-Means Clustering')
      hierarchicalClustering_img = ImageOps.expand(Image.open('img/user_manual/hierarchicalClustering.png'), border= 5, fill='white')
      st.image(hierarchicalClustering_img, caption='Hierarchical K-Means Clustering')
      st.markdown('<p style="text-align: justify;">It is a combination of two algorithms, namely Hierarchical Clustering and K-Means Clustering. By clicking the "Hierarchical K-Means Clustering" button, the application will automatically perform computations starting from validation using Sillhouette Score, data visualization, to determining output results based on RFM mapping.</p>', unsafe_allow_html=True)
      hierarchicalClustering_after_img = ImageOps.expand(Image.open('img/user_manual/hierarchicalClustering_after.png'), border= 5, fill='white')
      st.image(hierarchicalClustering_after_img, caption='After Clicking Hierarchical K-Means Clustering')

      # Show Data
      st.subheader('1.5. Show Data')
      showDataset_img = ImageOps.expand(Image.open('img/user_manual/showDataset.png'), border= 5, fill='white')
      st.image(showDataset_img, caption='Show Data')
      st.markdown('<p style="text-align: justify;">by clicking the "Show Data" checkbox, the user can see the preprocessed and clustered data</p>', unsafe_allow_html=True)
      showDatasetAfter_img = ImageOps.expand(Image.open('img/user_manual/showDatasetAfter.png'), border= 5, fill='white')
      st.image(showDatasetAfter_img, caption='Show Data Result')

      # Save Data
      st.subheader('1.6. Save Data')
      saveDataset_img = ImageOps.expand(Image.open('img/user_manual/saveDataset.png'), border= 5, fill='white')
      st.image(saveDataset_img, caption='Show Data')
      st.markdown('<p style="text-align: justify;">by clicking the "Save Data" button, the user can download the preprocessed and clustered data</p>', unsafe_allow_html=True)


    if menu == 'About':
      st.markdown("<h2 style='text-align: center;'>About the Application</h2>", unsafe_allow_html=True)
      st.markdown('<p style="text-align: justify;">This application is a simulation program for customer segmentation calculations based on a combination of Hierarchical Clustering and K-Means Clustering methods adjusted to Recency, Frequency, and Monetary models.</p>', unsafe_allow_html=True)
      st.markdown('<p style="text-align: justify;">The flowchart of the combination of these algorithms can be seen below</p>', unsafe_allow_html=True)
      flowchart_hkMeans_img = ImageOps.expand(Image.open('img/user_manual/flowchart_hkMeans.png'), border= 5, fill='white')
      st.image(flowchart_hkMeans_img, caption='Flowchart Algorithm')

      st.markdown('<p style="text-align: justify;">This application was made for the faculty of informatics engineering at Tarumanagara University in order to meet the needs of a thesis entitled "Penerapan Data Mining Menggunakan K-Means dan Hierarchical Clustering Terhadap Data Retail Online Berdasarkan Analisa Recency, Frequency, dan Monetary (RFM)" </p>', unsafe_allow_html=True)


      st.markdown("<h2 style='text-align: center;'>About the Author</h2>", unsafe_allow_html=True)
      st.markdown('<p style="text-align: justify;">Andreas lie, is currently a student at Tarumanagara University. Majoring in Informatics, there he studied Data Engineering and made this application using knowledge gained from those subject.</p>', unsafe_allow_html=True)
      st.markdown('<p style="text-align: justify;">You can contact him through email : andreaslie19@gmail.com</p>', unsafe_allow_html=True)





# Hide Streamlit markdown
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == '__main__':
	main()



