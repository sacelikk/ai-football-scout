import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Football Scout", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .stMetric {
        background-color: rgba(29, 185, 84, 0.05); 
        padding: 10px; 
        border-radius: 10px; 
        border-left: 4px solid #1DB954;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(#1DB954, #121212);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stContainer {
        padding: 10px;
        border-radius: 10px;
        background-color: #fcfcfc;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 AI Odaklı Küresel Scout Platformu")
st.markdown("Daha fazla istatistik ve derinlemesine özellik mühendisliği (Feature Engineering) ile güçlendirilmiş yetenek avcısı.")
st.markdown("---")

# İstatistiklerin Türkçe çevirileri
stat_isimleri = {
    'Gls': 'Gol',
    'Ast': 'Asist',
    'Sh': 'Şut',
    'SoT': 'İsab. Şut',
    'TklW': 'Top Kapma',
    'Int': 'Pas Arası',
    'Crs': 'Orta Açma',
    'Fld': 'Faul Alma (Dribbling)',
    'Fls': 'Agresiflik (Faul)'
}

@st.cache_data
def tum_veriyi_hazirla():
    def sezonu_isle(dosya_adi, etiket):
        df = pd.read_csv(dosya_adi)
        df = df[df['Min'] >= 500].copy()
        df['90s'] = pd.to_numeric(df['90s'], errors='coerce').fillna(1)
        df = df[df['90s'] > 0]
        
        # YENİ İSTATİSTİKLER EKLENDİ (Orta açma, faul alma, faul yapma vb.)
        tum_istatistikler = ['Gls', 'Ast', 'Sh', 'SoT', 'TklW', 'Int', 'Crs', 'Fld', 'Fls']
        for stat in tum_istatistikler:
            df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
            df[f'{stat}_per90_{etiket}'] = df[stat] / df['90s']
            
        lazim = ['Player', 'Squad', 'Comp', 'Pos'] + [f'{stat}_per90_{etiket}' for stat in tum_istatistikler]
        return df[lazim]

    try:
        df1 = sezonu_isle("players_data-2024_2025.csv", "Eski")
        df2 = sezonu_isle("players_data-2025_2026.csv", "Yeni")
    except:
        return pd.DataFrame()
    
    df = pd.merge(df1, df2, on='Player', how='inner', suffixes=('_eski', '_yeni'))
    df['Squad'] = df['Squad_yeni']
    df['Comp'] = df['Comp_yeni']
    df['Pos'] = df['Pos_yeni']
    
    tum_istatistikler = ['Gls', 'Ast', 'Sh', 'SoT', 'TklW', 'Int', 'Crs', 'Fld', 'Fls']
    for stat in tum_istatistikler:
        df[f'{stat}_farki'] = df[f'{stat}_per90_Yeni'] - df[f'{stat}_per90_Eski']
        
    lig_agirliklari = {'Premier League': 1.0, 'La Liga': 0.95, 'Serie A': 0.95, 'Bundesliga': 0.90, 'Ligue 1': 0.85}
    def katsayi_bul(lig):
        if not isinstance(lig, str): return 0.7
        for l, k in lig_agirliklari.items():
            if l in lig: return k
        return 0.7
        
    df['Lig_Katsayisi'] = df['Comp'].apply(katsayi_bul)
    
    return df.reset_index(drop=True)

df_genel = tum_veriyi_hazirla()

if len(df_genel) == 0:
    st.error("Veri seti bulunamadı.")
    st.stop()

st.sidebar.header("🔍 Scout Filtreleri")

mevki_sozlugu = {
    "Tüm Mevkiler": "",
    "Forvetler (Hücum)": "FW",
    "Orta Sahalar (Merkez)": "MF",
    "Defanslar (Savunma)": "DF"
}
secilen_mevki_etiket = st.sidebar.selectbox("Aranacak Mevkiyi Seçin:", list(mevki_sozlugu.keys()))
mevki_kodu = mevki_sozlugu[secilen_mevki_etiket]

if mevki_kodu:
    df_filtrelenmis = df_genel[df_genel['Pos'].str.contains(mevki_kodu, na=False)].reset_index(drop=True)
else:
    df_filtrelenmis = df_genel.copy()

# OYUNCU POZİSYONUNA GÖRE YENİ VE DERİNLEMESİNE İSTATİSTİKLER
if mevki_kodu == "FW":
    # Fld: Kendisine faul yapılması (Tehlike yaratan, tutulamayan forvetleri bulur)
    aktif_istatistikler = ['Gls', 'Ast', 'Sh', 'SoT', 'Fld'] 
elif mevki_kodu == "MF":
    # Crs: Orta açma (Oyun kurucu ve kanat özellikleri)
    aktif_istatistikler = ['Ast', 'Crs', 'Int', 'TklW', 'Gls']
elif mevki_kodu == "DF":
    # Fls: Faul yapma (Agresiflik, sert savunma göstergesi)
    aktif_istatistikler = ['TklW', 'Int', 'Crs', 'Fls']
else:
    aktif_istatistikler = ['Gls', 'Ast', 'Crs', 'TklW', 'Int']

nihai_ozellikler = []
for stat in aktif_istatistikler:
    df_filtrelenmis[f'{stat}_yeni_ag'] = df_filtrelenmis[f'{stat}_per90_Yeni'] * df_filtrelenmis['Lig_Katsayisi']
    df_filtrelenmis[f'{stat}_gelisim_ag'] = df_filtrelenmis[f'{stat}_farki'] * df_filtrelenmis['Lig_Katsayisi']
    nihai_ozellikler.extend([f'{stat}_yeni_ag', f'{stat}_gelisim_ag'])

if len(df_filtrelenmis) > 5:
    scaler = StandardScaler()
    olcekli_veri = scaler.fit_transform(df_filtrelenmis[nihai_ozellikler])
    model = NearestNeighbors(n_neighbors=6, algorithm='auto')
    model.fit(olcekli_veri)

    oyuncu_listesi = df_filtrelenmis['Player'].sort_values().tolist()
    secilen_oyuncu = st.sidebar.selectbox("Hedef Oyuncuyu Seçin:", oyuncu_listesi)

    if secilen_oyuncu:
        hedef_idx = df_filtrelenmis[df_filtrelenmis['Player'] == secilen_oyuncu].index[0]
        hedef = df_filtrelenmis.iloc[hedef_idx]
        
        st.subheader(f"🎯 Hedef Profil: {hedef['Player']}")
        st.caption(f"Takım: **{hedef['Squad']}** | Lig: **{hedef['Comp']}** | Pozisyon: **{hedef['Pos']}**")
        
        cols = st.columns(len(aktif_istatistikler))
        for i, stat in enumerate(aktif_istatistikler):
            cols[i].metric(f"{stat_isimleri[stat]} (Maç Başı)", f"{hedef[f'{stat}_per90_Yeni']:.2f}", f"{hedef[f'{stat}_farki']:.2f} trend")

        st.markdown("<br><h3>🤖 Makine Öğrenmesinin Gelişmiş Önerileri</h3>", unsafe_allow_html=True)
        
        mesafeler, benzerler = model.kneighbors([olcekli_veri[hedef_idx]])
        
        for i in range(1, len(benzerler[0])):
            idx = benzerler[0][i]
            skor = mesafeler[0][i]
            onerilen = df_filtrelenmis.iloc[idx]
            
            with st.container():
                st.markdown(f"#### {i}. {onerilen['Player']} | {onerilen['Squad']}")
                st.caption(f"Lig: **{onerilen['Comp']}** | KNN Benzerlik Puanı: **{skor:.2f}**")
                
                c_list = st.columns(len(aktif_istatistikler))
                for j, stat in enumerate(aktif_istatistikler):
                    c_list[j].metric(f"{stat_isimleri[stat]}", f"{onerilen[f'{stat}_per90_Yeni']:.2f}", f"{onerilen[f'{stat}_farki']:.2f}")
                st.markdown("---")
else:
    st.warning("Bu filtrelerde makine öğrenmesi analizi yapacak yeterli oyuncu bulunamadı.")
