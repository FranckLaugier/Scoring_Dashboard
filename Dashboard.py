import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import plotly.graph_objects as go
import shap

# URL API : local et cloud
# URL_API = "http://127.0.0.1:5000/"
URL_API = 'https://franck-app-heroku.herokuapp.com/'

st.set_page_config(layout="wide") # Affichage large

# -------------------------------------------------------------------------------
# ########################## Import des données #################################
#--------------------------------------------------------------------------------
def pickle_load(file):
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data

# Les données
X_test = pd.read_csv('./X_test.csv')

# Les features
features_70 = pickle_load('liste_70_feature_importances.pickle')
liste_lineplot = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'PAYMENT_RATE', 'YEAR_BIRTH',
                  'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE', 'INCOME_PER_PERSON']
liste_hist = ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE']
features_to_compare = liste_lineplot + liste_hist

# Les données de densité
dico_kde = pickle_load('dictionnaire_kde.pickle')
# Les données des histogrammes
dico_hist = pickle_load('dictionnaire_hist.pickle')
df_hist =  pickle_load('df_hist_export.pickle')

# Les shap_values
shap_values = pickle_load('shap_values.pickle')
exp_values = pickle_load('exp_values.pickle')

# La liste des clients à prédire
liste_client_ID = X_test['SK_ID_CURR'].values
# Transformation des variables catégorielles
X_test['CODE_GENDER'] = X_test['CODE_GENDER'].map({0: 'M',
                                                   1: 'F'})
X_test['FLAG_OWN_CAR'] = X_test['FLAG_OWN_CAR'].map({0: 'N',
                                                     1: 'Y'})
X_test['FLAG_OWN_REALTY'] = X_test['FLAG_OWN_REALTY'].map({0: 'N',
                                                           1: 'Y'})
X_test['YEAR_BIRTH'] = -X_test['DAYS_BIRTH'] / 365

logo_ocr = imread("./logo_OCR.png")
logo_pret = imread("./logo_pret.png")

# -------------------------------------------------------------------------------
# ########################## Définition du main () ##############################
#--------------------------------------------------------------------------------
def main():
    st.sidebar.image(logo_pret)

    PAGES = ["Accueil", "Visualisation score", "Comparaison clientèle"]

    st.sidebar.write('')
    st.sidebar.write('')

    st.sidebar.title('Pages')
    selection = st.sidebar.radio(" ", PAGES)

    if selection == "Accueil":
        init = accueil()
    if selection == "Visualisation score":
        prediction, index_client, id_client = infos_generales_client()  # Affichage des informations client
        score_viz(prediction, exp_values, index_client)  # Affichage de la prédiction sur le compteur
        dataframe_client(id_client)
    if selection == "Comparaison clientèle":
        comparaison_client()

# -------------------------------------------------------------------------------
# ########### Définition de la page accueil et fonctions générales ##############
#--------------------------------------------------------------------------------
def accueil():
    # Affichage du logo et du titre
    col1, col2 = st.columns([1,6])
    with col1:
        st.image(logo_ocr)
    with col2:
        st.title("P7 - Implémenter un modèle de scoring")

    st.write('')
    st.write('')
    st.write('')
    st.write('Dans le cardre du parcours Data Scientist, nous travaillons pour l’entreprise : "Prêt à dépenser".')
    st.write('')

    mystyle = '''
        <style>
            p {
                text-align: justify;
            }
        </style>
        '''

    st.markdown(mystyle, unsafe_allow_html=True)
    st.write('L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit en s’appuyant' 
    ' sur des sources de données variées (données comportementales, données provenant d’autres institutions financières, etc.).'
    ' Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent'
    ' à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, '
    ' mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. ')

    st.header('Spécifications du dashboard')
    st.write(' - Permettre de visualiser le score et l’interprétation de ce score pour chaque client')
    st.write(' - Permettre de visualiser des informations descriptives relatives à un client')
    st.write(' - Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients')

    st.write('')
    st.write('')
    st.write('')
    st.write('Le calcul de la probabilité doit être réalisé via une API dans le cloud.')

def load_prediction(id_client):
    # Requête permettant de récupérer la prédiction de faillite du client sélectionné
    print(id_client)
    pred = requests.get(URL_API + "prediction/" + str(id_client), params={"id_client": id_client})
    print(pred)
    pred = pred.json()
    prediction = pred['prediction'][1]
    print(prediction)

    if prediction > 0.495:
        decision = "Rejeté"
    else:
        decision = "Approuvé"

    return prediction, decision

def color(prediction):
    '''Définition de la couleur selon la prédiction pour le compteur'''
    if prediction < 0.495:
        col = 'green'
    else:
        col = 'red'
    return col

def color_compteur(pred):
    if pred < 0.495:
        dico_color_1 = {'range': [0.05, 0.495], 'color': 'lightgreen'}
        dico_color_2 = {'range': [0.495, 0.92], 'color': 'whitesmoke'}
    else:
        dico_color_1 = {'range': [0.05, 0.495], 'color': 'whitesmoke'}
        dico_color_2 = {'range': [0.495, 0.92], 'color': 'lightcoral'}
    return dico_color_1, dico_color_2

def titre_centre(text_titre):
    st.write('')
    st.write('')
    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write('')
    with col2:
        st.markdown("""
                    <h1 style='text-align: center'>{}</h1>
                    """.format(text_titre), unsafe_allow_html=True)
    with col3:
        st.write('')

def find_status(id_client, col):
    """Fonction permettant de trouver le status marital à partir de plusieurs colonnes OHE"""
    df_client = X_test[X_test['SK_ID_CURR'] == id_client]
    filter_col = [columns for columns in X_test if columns.startswith(str(col))] # Permet de ne garder que les colonnes qui commencent par la feature ex :"NAME_FAMILT_STATUS_xxxx"
    df_client = df_client[filter_col]
    fam_status = df_client.columns[(df_client == 1).any(axis=0)] # retourne le nom de la seule colonne qui contient 1 dans la ligne (axis=0)
    return fam_status[0].split('_')[-1]  # Après OHE, les noms sont de la forme NAME_FAMIMLY_STATUS_XXXX, permet de récupérer XXXX

# -------------------------------------------------------------------------------
# ################# Définition de la page visualisation score ###################
#--------------------------------------------------------------------------------
####################### Fonction infos clients ##################################
def find_info_client(id_client):
    ''' Fonction permettant de récupérer les infos clients dans un dictionnaire '''
    df_client = X_test[X_test['SK_ID_CURR'] == id_client]
    dict_infos_client = {}
    dict_infos_client = {
        "genre" : df_client["CODE_GENDER"].item(),
        "proprietaire_voiture" : df_client["FLAG_OWN_CAR"].item(),
        "proprietaire_bien" : df_client["FLAG_OWN_REALTY"].item(),
        "status_famille" : find_status(id_client, "NAME_FAMILY_STATUS"),
        "nb_enfant" : df_client["CNT_CHILDREN"].item(),
        "age" : int(df_client["DAYS_BIRTH"].values / -365),
        "revenus" : int(df_client["AMT_INCOME_TOTAL"].item()),
        "revenus par personne du foyer" : int(df_client["INCOME_PER_PERSON"].item()),
        "montant_credit" : int(df_client["AMT_CREDIT"].item()),
        "annuites" : int(df_client["AMT_ANNUITY"].item()),
        "montant_bien" : int(df_client["AMT_GOODS_PRICE"].item())
       }
    return dict_infos_client

def infos_generales_client():
    '''Fonction permettant de récupérer et afficher les infos clients'''
    # Chargement de la selectbox et récupération id_client
    id_client = st.sidebar.selectbox("Sélection du client", liste_client_ID)
    # Calcul index_client
    index_client = X_test.index[X_test['SK_ID_CURR'] == id_client][0]

    # Récupérations des infos clients
    dict_infos_client = find_info_client(id_client)
    # Réalisation prédiction
    prediction, decision = load_prediction(id_client)

    st.sidebar.markdown(' ')
    titre_centre("Dashboard données client")
    st.write('')
    st.markdown("<u><b>Décision finale :</b></u>", unsafe_allow_html=True)

    # Affichage de la décision dans la couleur
    if decision == 'Rejeté':
        st.error(decision)
    else:
        st.success(decision)

    st.markdown(' ')
    st.markdown("<u><b>Profil client :</b></u>", unsafe_allow_html=True)

    # Définition des positions des textes
    c1, c2, c3, c4, c5 = st.columns(5)
    # Ecriture des infos clients
    with st.container():
        c1.write('**ID client** : '+str(id_client))
        c2.write('**Sexe** : ' + str(dict_infos_client['genre']))
        c3.write('**Statut familial** : ' + str(dict_infos_client['status_famille']))
        c4.write("**Nombre d'enfants** : " + str(dict_infos_client['nb_enfant']))
        c5.write('**Age** : ' + str(dict_infos_client['age']))

    st.markdown(' ')
    c1, c2, c3, c4, c5 = st.columns(5)
    with st.container():
        c1.write('**Propriétaire voiture** : '+str(dict_infos_client['proprietaire_voiture']))
        c2.write('**Propriétaire logement** : ' + str(dict_infos_client['proprietaire_bien']))
        c3.write('**Revenus** : ' + str(dict_infos_client['revenus']) + " $")
        c4.write("**Montant crédit** : " + str(dict_infos_client['montant_credit']) + " $")
        c5.write('**Annuités** : ' + str(dict_infos_client['annuites']) + " $")

    return prediction, index_client, id_client

########################### Fonction compteur #######################################
def score_viz(pred, exp_values, index_client):
    ''' Fonction permettant la visualisation du compteur'''
    titre_centre('Visualisation score')

    # Pour centrer le graph
    col1, col2, col3 = st.columns([3,1,6])
    with col1:
        for i in range(10):
            st.write('')

        st.success('Valeur en dessous du seuil = financement accordé')
        st.write('')
        st.write('')
        st.error('Valeur au dessus du seuil = financement refusé')

    with col2:
        st.write(' ')

    with col3:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pred,
            number = {'font':{'size':48}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            # title = {'text': str(round(pred*100, 2)) + "%", 'font': {'size': 28, 'color':color(pred)}},
            title={'text': "Valeur seuil à 0.495", 'font': {'size': 28, 'color': 'black'}},
            delta = {'reference': 0.495, 'increasing': {'color': "red"},'decreasing':{'color':'green'}},
            gauge = {
                'axis': {'range': [0,1], 'tickcolor': color(pred)},
                'bar': {'color': color(pred)},
                'steps': [
                    {'range' : [0, 0.05], 'color': 'black'},
                    color_compteur(pred)[0],
                    color_compteur(pred)[1],
                    {'range': [0.92, 1], 'color': 'black'}],
                'threshold': {
                    'line': {'color': "black", 'width': 5},
                    'thickness': 1,
                    'value': 0.495}}))
        st.plotly_chart(fig)


    #Affichage du force_plot shap
    titre_centre('Features importance')

    st_shap(shap.force_plot(exp_values, shap_values[index_client], features=X_test[features_70].iloc[index_client],
                            feature_names=features_70, figsize=(25,5)))

def st_shap(plot, height=None):
    """Fonction permettant l'affichage de graphique shap values"""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

########################### Dataframe des données client #######################################
def dataframe_client(id_client):
    titre_centre('Données client')

    df_client = pd.concat([X_test.drop(['SK_ID_CURR'], axis=1), df_hist], axis=1)
    df_client = df_client[df_client['SK_ID_CURR'] == id_client]
    choix = st.radio('Options de visualisation :', ['Données simplifiées', 'Données complètes'])

    if choix == 'Données simplifiées':
        st.dataframe(df_client[features_to_compare])
    else:
        st.dataframe(df_client[features_70])

# -------------------------------------------------------------------------------
####################### Définition de la page comparaison #######################
#--------------------------------------------------------------------------------
def comparaison_client():
    '''Fonction pour la page comparaison client'''
    titre_centre("Dashboard données client")
    st.markdown("<u><b>Comparaison clientèle :</b></u>", unsafe_allow_html=True)

    # Chargement de la selectbox
    id_client = st.sidebar.selectbox("Sélection du client", liste_client_ID)
    st.sidebar.markdown(' ')

    index_client = X_test.index[X_test['SK_ID_CURR'] == id_client][0]

    # multi select box
    features = st.multiselect("Données à comparer : ", features_to_compare, default=['NAME_EDUCATION_TYPE',
                                                                                               'AMT_CREDIT',
                                                                                               'AMT_INCOME_TOTAL',
                                                                                               'NAME_INCOME_TYPE',
                                                                                               'YEAR_BIRTH',
                                                                                               'AMT_GOODS_PRICE'
                                                                                               ])
    st.write(' ')
    if st.button('Montrer tout'):
        features = features_to_compare

    # Définition des positions des graphs, 3 colonnes, 12 graph max
    pos1_1, pos1_2, pos1_3 = st.columns(3)
    st.write('')
    pos2_1, pos2_2, pos2_3 = st.columns(3)
    st.write('')
    pos3_1, pos3_2, pos3_3 = st.columns(3)
    st.write('')
    pos4_1, pos4_2, pos4_3 = st.columns(3)
    st.write('')
    pos5_1, pos5_2, pos5_3 = st.columns(3)

    liste_pos = [pos1_1, pos1_2, pos1_3, pos2_1, pos2_2, pos2_3,
                 pos3_1, pos3_2, pos3_3, pos4_1, pos4_2, pos4_3,
                 pos5_1, pos5_2, pos5_3]

    # Réalisation des graph à la bonne position
    for f,pos in zip(features, liste_pos):
        graphiques(f, pos, index_client, id_client)

########################## Fonctions graphiques ################################
def graphiques(col, pos, index_client, id_client):
    """Affichage des graphes de comparaison pour le client sélectionné """
    liste_log = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'INCOME_PER_PERSON', 'AMT_ANNUITY']

    if col in(liste_hist):
        graph_hist(str(col), pos, col, index_client, id_client)
    else:
        if col not in liste_log:
            graph_kde(str(col), pos, col, index_client)
        else:
            graph_kde_log(str(col), pos, col, index_client)

def graph_kde(title, pos, col, index_client):
    """Définition des graphes KDE avec une ligne verticale indiquant la position du client"""
    with pos:
        st.subheader(title)
        fig, ax = plt.subplots()
        sns.lineplot(dico_kde[col][0], dico_kde[col][1], color='green', label='Financement accepté')
        sns.lineplot(dico_kde[col][2], dico_kde[col][3], color='red', label='Financement refusé')
        plt.axvline(x=X_test.loc[index_client, col], ymax=0.95, label='Client', linewidth=2, linestyle=("--"), color='black')
        plt.legend()
        st.pyplot(fig)

def graph_kde_log(title, pos, col, index_client):
    """Définition des graphes KDE avec une ligne verticale indiquant la position du client"""
    with pos:
        st.subheader(title)
        fig, ax = plt.subplots()
        sns.lineplot(dico_kde[col][0], dico_kde[col][1], color='green', label='Financement accepté')
        sns.lineplot(dico_kde[col][2], dico_kde[col][3], color='red', label='Financement refusé')
        plt.axvline(x=X_test.loc[index_client, col], ymax=0.95, label='Client', linewidth=2, linestyle=("--"), color='black')
        plt.xscale('log')
        plt.legend()
        st.pyplot(fig)

def graph_hist(title, pos, col, index_client, id_client):
    '''Définition des histogrammes'''
    with pos:
        st.subheader(title)
        fig, ax = plt.subplots(figsize=(7, 5.3))
        df = pd.DataFrame(dico_hist[col]) # Trie pour n'avoir que la feature sélectionnée
        col_max = df.max().idxmax()  # Récupération de la colonne TARGET 0 et 1 où est le max
        df = df.sort_values([col_max], ascending=False) # Trie sur la colonne où il y a le max
        df = df.head(5)  # Conservation que des 5 histogrammes max
        df_temp = df.reset_index() # Reset_index pour mettre l'index en colonne
        df_temp.rename(columns={'index': col}, inplace=True) # Remplace la nouvelle colonne 'index' par le nom de la feature
        df_temp = df_temp.melt(id_vars=[col], var_name='TARGET')
        palette = ['green', 'red']
        sns.barplot(y='value', x=col, hue='TARGET', data=df_temp, palette=palette)
        value_client = df_hist[df_hist['SK_ID_CURR'] == id_client][col].iloc[0] # Récupération de la valeur de l'histogramme pour la feature donnée
        # Pour le x il faut un nombre (0, 1, 2, ...) pour positionner la barre sur le barplot
        plt.axvline(x=df.index.get_loc(value_client), ymax=0.95, label='Client', linewidth=4, linestyle=("--"), color='black')
        plt.yscale('log')
        plt.xlabel('')
        plt.xticks(rotation=75)
        plt.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
