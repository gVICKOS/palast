import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from PIL import Image


def main():
    #st.image("Logo-HD-eni-EGS.jpg", width=100, caption="Your Picture Caption")
    image = Image.open('Logo-HD-eni-EGS.png')
    st.image(image, caption=None, width=250, use_column_width=None, clamp=False, channels="RGB")
    st.title ("Prédiction des Plans d'Apurement")
    #st.subheader("Autres infos")
    st.markdown("Cette application utilise un modèle de Machine Learning pour prédire la tranche de PA à proposer au client en fonction de ses caractéristiques.")
    
    ## Fonction d'importation des données
    @st.cache_data
    def load_data(): 
        data = pd.read_excel("DataBase22Ter.xlsx")
        data.drop(data.columns[[2,3,7,8,20,21,23]], axis=1, inplace=True)
        datapart= data[data.Segment== "RESIDENT"]
        datapart['Mode_envoi']= datapart['Mode_envoi'].replace('0', np.nan, regex=True)
        datapart = datapart.loc[datapart.Mt_versement <=2500]
        datapart = datapart.loc[datapart.Mt_versement > 20 ]
        datapart = datapart.loc[datapart.Montant> 0]
        datapart = datapart.loc[datapart.Montant< 1750]
        dataclean= datapart.dropna()
        dataclean.drop(dataclean.columns[[2,5,10,11,14,15,16]], axis=1, inplace=True)
        dataclean['Contrat6Mois']= dataclean['Contrat6Mois'].replace('Oui', 'Yes', regex=True)
        dataclean['Contrat6Mois']= dataclean['Contrat6Mois'].replace('Non', 'No', regex=True)
        return dataclean
    ## Affichage de la table de données 
    df= load_data()
    
    df_sample = df.sample(50) 
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader ("Jeu de données: Échantillon de 50 Observations")
        st.write(df_sample)
    
  #  seed= 123
    
    #Train/test Split 
   # def split (df): 
    #    yT = df.Mt_versement
    #    X = df.loc[:,df.columns.difference(['Mt_versement'])]
    #    y = pd.cut(yT,yT.quantile([0,.25,.50,.75,1]), include_lowest=True)
     #   y = y.loc[yT<=2500]
    #    X_train, X_test, y_train, y_test = train_test_split(
     #       X,y,
      #      test_size= 0.3,
       #     stratify=y,
     #       random_state=seed
     #   )
        
  #  X_train, X_test, y_train, y_test = train_test_split(df)
        
    # Collecter le profil d'entrée
    st.sidebar.header("Les caractéristiques du client")
    st.sidebar.markdown("*Choisir les informations relatives au contrat*")
    def client_caract_entree():
        Offre6Mois= st.sidebar.selectbox('Offre expirant dans 6 mois au plus',('Oui','Non'))
        Contrat6Mois=st.sidebar.selectbox('Contrat expirant dans 6 mois au plus',('Yes','No'))
        Montant_PA= st.sidebar.number_input('Montant à régler', value=200)
        Mode_envoi= st.sidebar.selectbox("Mode d'envoi",('Imprimante standard','Envoyer un e-mail'))
        Délais_paiement= st.sidebar.selectbox('Délais de paiement',('Paiement le 5 du mois','Paiement dans 15 jours','Paiement le 15 du mois','Paiement le 25 du mois','Paiement le 10 du mois','Paiement le 1 du mois '))
        Mode_paiement= st.sidebar.selectbox('Mode de paiement',('Prelevement','Cheque','Especes en mandat'))
        Statut= st.sidebar.selectbox('Statut',('Dual','Mono E','Mono G'))
        Cycle_fact= st.sidebar.selectbox('Cycle de facturation',('Echeancier','Bimestriel'))
        Montant=st.sidebar.number_input('Montant moyen de factures', value= 50)
        
        data= {
        'Offre6Mois': Offre6Mois,
        'Contrat6Mois':Contrat6Mois,
        'Montant_PA': Montant_PA,
        'Mode_envoi': Mode_envoi,
        'Délais_paiement': Délais_paiement,
        'Mode_paiement': Mode_paiement,
        'Statut': Statut,
        'Cycle_fact': Cycle_fact,
        'Montant': Montant
        }
        profil_client= pd.DataFrame(data,index=[0])
        return profil_client
    input_df= client_caract_entree()
        
        
    # Transformer la donnée en donnée adaptée au modèle 
    #importer les données
    my_input = df.drop(['Mt_versement'], axis=1)
    donnee_entree=pd.concat([input_df,my_input],axis=0)
    
    # encodage des données
    #var_cat= ['Offre6Mois','Contrat6Mois','Mode_envoi','Délais_paiement','Mode_paiement','Statut','Cycle_fact']
   # for col in var_cat: 
   #     dummy= pd.get_dummies(donnee_entree[col], drop_first=True)
   #     donnee_entree= pd.concat([dummy,donnee_entree], axis=1)
   #     del donnee_entree[col]
   # df_cat = donnee_entree.select_dtypes(include=['object'])
   # df_int = donnee_entree.select_dtypes(exclude=['object'])
    num_cols = donnee_entree.select_dtypes(exclude=["object"]).columns
    std_scaler = preprocessing.StandardScaler()
    std_scaler.fit(donnee_entree.loc[:,num_cols])
    donnee_entree.loc[:,num_cols] = std_scaler.transform(donnee_entree.loc[:,num_cols])
    data_enc = pd.get_dummies(donnee_entree, drop_first=True)
    # prendre uniquement la première ligne 
    donnee_finale= data_enc[:1]
    
    # Afficher les données transformées 
    #st.subheader('Les données transformées')
    #st.write(donnee_finale)
    st.markdown("*Choisissez les caractéristiques du client ensuite cliquez sur le bouton Prédire.*") 
        
    # Importer le modèle 
    load_model= joblib.load('model.joblib')
    
    # Appliquer le modèle sur les données en entrée 
    if st.button("Prédire"):
        prevision= load_model.predict(donnee_finale)
        rounded = [float(np.round(x)) for x in prevision]
        my_float = float(rounded[0])
        resultat= "Le montant à proposer pour ce client est égal à : " + str(my_float) + "€"
        st.success(resultat)
        re= input_df.iloc[:,2] / my_float
        rd = [float(np.round(x)) for x in re]
        my_fl = float(rd[0])
        #nb= "nombre" + str(round(re),0)
        result= "Le nombre de versements à proposer est égal à : " + str(my_fl) 
        st.success(result)
        st.markdown("Le taux de fiabilité de ce modèle est de 60%.") 
    #st.subheader('Résultats de la prévision')
    #st.write(prevision)
if __name__ == '__main__': 
    main()


