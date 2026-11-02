"""
🏗️ Analyse des Caractéristiques du Béton dans les Ouvrages
Version Python (Streamlit) de l'application R Shiny de NGOUALEM CLYDE 22P311

Installation :  pip install streamlit pandas numpy plotly
Lancement    :  streamlit run app_beton.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Analyse du Béton",
    page_icon="🏗️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
TYPES_OUVRAGE = ["Poteau", "Poutre", "Dalle", "Fondation", "Voile",
                 "Escalier", "Chape", "Autre"]
CLASSES = ["C16/20", "C20/25", "C25/30", "C30/37", "C35/45",
           "C40/50", "C45/55", "C50/60"]
TYPES_ESSAI = ["Compression", "Traction", "Flexion",
               "Module d'élasticité", "Durabilité", "Complet"]

CONFORME = "✅ Conforme"
ACCEPTABLE = "🟡 Acceptable"
NON_CONFORME = "❌ Non conforme"
COULEURS_STATUT = {
    CONFORME: "#27ae60",
    ACCEPTABLE: "#f39c12",
    NON_CONFORME: "#e74c3c",
}

SEUILS_DEFAUT = {
    "resistance_min": 25.0,
    "tolerance_affaissement": 2.0,
    "module_min": 30.0,
}


# ----------------------------------------------------------------------------
# Fonctions de calcul
# ----------------------------------------------------------------------------
def calcul_resistance_caracteristique(resistances):
    """fck = moyenne - 1,48 x écart-type (n >= 3), sinon minimum."""
    r = pd.Series(resistances).astype(float)
    if len(r) >= 3:
        return r.mean() - 1.48 * r.std(ddof=1)
    return r.min()


def verifier_conformite(resistance, resistance_min, affaissement,
                        affaissement_cible, tolerance, module, module_min):
    ok_res = resistance >= resistance_min
    ok_aff = abs(affaissement - affaissement_cible) <= tolerance
    ok_mod = module >= module_min
    if ok_res and ok_aff and ok_mod:
        return CONFORME
    if resistance >= resistance_min * 0.9 and ok_mod:
        return ACCEPTABLE
    return NON_CONFORME


def prediction_resistance_long_terme(resistance_28j, age_jours, type_ciment="CEM II"):
    """fcm(t) = fcm x exp[s x (1 - sqrt(28/t))]"""
    s = 0.25 if type_ciment == "CEM I" else 0.20
    if age_jours <= 28:
        return resistance_28j
    return resistance_28j * np.exp(s * (1 - np.sqrt(28 / age_jours)))


def coefficient_variation(resistances):
    r = pd.Series(resistances).astype(float)
    if len(r) < 2 or r.mean() == 0:
        return float("nan")
    return r.std(ddof=1) / r.mean() * 100


def fmt(x, nd=1, unite=""):
    if pd.isna(x):
        return "n/a"
    return f"{x:.{nd}f}{unite}"


# ----------------------------------------------------------------------------
# Données initiales et état de session
# ----------------------------------------------------------------------------
def donnees_initiales():
    return pd.DataFrame({
        "id": range(1, 7),
        "type_ouvrage": ["Poteau", "Poutre", "Dalle", "Fondation", "Voile", "Poteau"],
        "nom_ouvrage": ["Poteau P1", "Poutre G12", "Dalle RDC", "Semelle S1",
                        "Voile V2", "Poteau P3"],
        "date_fabrication": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10",
                                            "2024-01-15", "2024-01-20", "2024-01-25"]),
        "date_essai": pd.to_datetime(["2024-01-29", "2024-02-02", "2024-02-07",
                                      "2024-02-12", "2024-02-17", "2024-02-22"]),
        "age_essai": [28] * 6,
        "classe_resistance": ["C25/30", "C30/37", "C25/30", "C20/25", "C30/37", "C25/30"],
        "resistance_compression": [32.5, 35.2, 28.8, 26.3, 36.7, 31.9],
        "resistance_traction": [2.8, 3.1, 2.5, 2.3, 3.2, 2.7],
        "module_elasticite": [33.2, 34.8, 32.1, 30.5, 35.2, 32.9],
        "affaissement": [9.5, 8.2, 12.3, 7.8, 10.5, 8.9],
        "masse_volumique": [2410, 2420, 2390, 2405, 2425, 2412],
        "type_essai": ["Compression", "Complet", "Compression", "Compression",
                       "Complet", "Compression"],
        "observations": ["Bon aspect", "Fissures fines", "Surface lisse",
                         "Granulométrie homogène", "Excellent compactage", "Légers nids"],
    })


if "donnees" not in st.session_state:
    st.session_state.donnees = donnees_initiales()
if "seuils" not in st.session_state:
    st.session_state.seuils = dict(SEUILS_DEFAUT)
if "date_essai" not in st.session_state:
    st.session_state.date_essai = (pd.Timestamp.today() + pd.Timedelta(days=28)).date()


def ajouter_essai():
    s = st.session_state
    nom = (s.nom_ouvrage or "").strip()
    if not nom:
        st.toast("❌ Renseignez le nom / la référence de l'ouvrage", icon="⚠️")
        return
    if s.resistance_compression <= 0:
        st.toast("❌ La résistance doit être positive", icon="⚠️")
        return

    df = s.donnees
    nouvel_id = int(df["id"].max()) + 1 if len(df) else 1
    nouvel_essai = pd.DataFrame([{
        "id": nouvel_id,
        "type_ouvrage": s.type_ouvrage,
        "nom_ouvrage": nom,
        "date_fabrication": pd.Timestamp(s.date_fabrication),
        "date_essai": pd.Timestamp(s.date_essai),
        "age_essai": s.age_essai,
        "classe_resistance": s.classe_resistance,
        "resistance_compression": s.resistance_compression,
        "resistance_traction": s.resistance_traction,
        "module_elasticite": s.module_elasticite,
        "affaissement": s.affaissement,
        "masse_volumique": s.masse_volumique,
        "type_essai": s.type_essai,
        "observations": s.observations,
    }])
    s.donnees = pd.concat([df, nouvel_essai], ignore_index=True)

    # Réinitialisation partielle
    s.nom_ouvrage = ""
    s.observations = ""
    st.toast("✅ Essai sur béton enregistré avec succès !")


def donnees_calculees():
    """Ajoute le statut de conformité, recalculé avec les seuils appliqués."""
    df = st.session_state.donnees.copy()
    sx = st.session_state.seuils
    if len(df):
        df["statut_conformite"] = [
            verifier_conformite(r, sx["resistance_min"], a, 8,
                                sx["tolerance_affaissement"], m, sx["module_min"])
            for r, a, m in zip(df["resistance_compression"], df["affaissement"],
                               df["module_elasticite"])
        ]
    else:
        df["statut_conformite"] = pd.Series(dtype=str)
    return df


# ----------------------------------------------------------------------------
# Barre latérale : saisie + paramètres de contrôle
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📥 Saisie des Données Béton")

    st.selectbox("Type d'ouvrage :", TYPES_OUVRAGE, key="type_ouvrage")
    st.text_input("Nom/Reference ouvrage :", placeholder="Ex: Poteau P1-RDC",
                  key="nom_ouvrage")
    st.date_input("Date de fabrication :", value=pd.Timestamp.today().date(),
                  key="date_fabrication", format="DD/MM/YYYY")
    st.date_input("Date de l'essai :", key="date_essai", format="DD/MM/YYYY")
    st.number_input("Âge du béton à l'essai (jours) :", min_value=1, max_value=365,
                    value=28, step=1, key="age_essai")
    st.selectbox("Classe de résistance :", CLASSES, key="classe_resistance")
    st.number_input("Résistance en compression (MPa) :", min_value=10.0,
                    max_value=100.0, value=30.0, step=0.1, key="resistance_compression")
    st.number_input("Résistance en traction (MPa) :", min_value=0.5, max_value=10.0,
                    value=2.5, step=0.1, key="resistance_traction")
    st.number_input("Module d'élasticité (GPa) :", min_value=10.0, max_value=50.0,
                    value=32.0, step=0.1, key="module_elasticite")
    st.number_input("Affaissement au cône (cm) :", min_value=0.0, max_value=20.0,
                    value=8.0, step=0.5, key="affaissement")
    st.number_input("Masse volumique (kg/m³) :", min_value=2000, max_value=2600,
                    value=2400, step=10, key="masse_volumique")
    st.selectbox("Type d'essai :", TYPES_ESSAI, key="type_essai")
    st.text_area("Observations :", placeholder="Conditions de cure, fissures, aspect...",
                 height=90, key="observations")
    st.button("✅ Ajouter Essai", type="primary", width="stretch",
              on_click=ajouter_essai)

    st.divider()
    st.subheader("⚙️ Paramètres de Contrôle")
    with st.form("form_seuils"):
        sx = st.session_state.seuils
        r_min = st.number_input("Résistance minimale requise (MPa) :", min_value=10.0,
                                max_value=60.0, value=float(sx["resistance_min"]), step=1.0)
        tol = st.number_input("Tolérance affaissement (± cm) :", min_value=0.5,
                              max_value=5.0, value=float(sx["tolerance_affaissement"]),
                              step=0.5)
        m_min = st.number_input("Module d'élasticité minimum (GPa) :", min_value=20.0,
                                max_value=40.0, value=float(sx["module_min"]), step=1.0)
        if st.form_submit_button("🔧 Appliquer Seuils", width="stretch"):
            st.session_state.seuils = {
                "resistance_min": r_min,
                "tolerance_affaissement": tol,
                "module_min": m_min,
            }
            st.rerun()


# ----------------------------------------------------------------------------
# Corps de la page
# ----------------------------------------------------------------------------
st.markdown(
    "<h2 style='color:#2c3e50;text-align:center;font-weight:bold;'>"
    "🏗️ Analyse des Caractéristiques du Béton dans les Ouvrages "
    "par NGOUALEM CLYDE 22P311</h2>",
    unsafe_allow_html=True,
)

df = donnees_calculees()
seuil_res = st.session_state.seuils["resistance_min"]


def carte(titre, valeur, sous_titre, couleur):
    st.markdown(
        f"""
        <div style="background:#ecf0f1;padding:15px;border-radius:5px;text-align:center;">
          <h4 style="color:{couleur};margin:0;">{titre}</h4>
          <h2 style="margin:6px 0;color:#2c3e50;">{valeur}</h2>
          <p style="margin:0;color:#555;">{sous_titre}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mise_en_forme(fig, hauteur=380):
    fig.update_layout(template="plotly_white", height=hauteur,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


tabs = st.tabs([
    "📊 Tableau de Bord",
    "🔍 Analyse Détaillée",
    "✅ Contrôle Qualité",
    "⏳ Durabilité",
    "📋 Base de Données",
    "📚 Guide Technique",
])

# ============================ Tab 1 : Tableau de bord =======================
with tabs[0]:
    if len(df):
        res_moy = fmt(df["resistance_compression"].mean(), 1, " MPa")
        taux = fmt((df["statut_conformite"] == CONFORME).mean() * 100, 1, "%")
        mod_moy = fmt(df["module_elasticite"].mean(), 1, " GPa")
        aff_moy = fmt(df["affaissement"].mean(), 1, " cm")
    else:
        res_moy, taux, mod_moy, aff_moy = "0 MPa", "0%", "0 GPa", "0 cm"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        carte("Résistance Moyenne", res_moy, "Compression (MPa)", "#2980b9")
    with c2:
        carte("Taux Conformité", taux, "Essais conformes", "#27ae60")
    with c3:
        carte("Module Élasticité", mod_moy, "Moyen (GPa)", "#e74c3c")
    with c4:
        carte("Affaissement Moyen", aff_moy, "Moyen (cm)", "#f39c12")

    st.write("")
    if len(df):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 📈 Évolution Résistance dans le Temps")
            d = df.sort_values("date_essai")
            fig = px.line(d, x="date_essai", y="resistance_compression",
                          color="type_ouvrage", markers=True,
                          hover_name="nom_ouvrage",
                          labels={"date_essai": "Date d'essai",
                                  "resistance_compression": "Résistance (MPa)",
                                  "type_ouvrage": "Ouvrage"})
            fig.add_hline(y=seuil_res, line_dash="dash", line_color="red")
            fig.update_layout(title="Évolution des résistances dans le temps")
            st.plotly_chart(mise_en_forme(fig), width="stretch")
        with g2:
            st.markdown("#### 🎯 Distribution des Résistances")
            fig = px.histogram(df, x="resistance_compression", color="statut_conformite",
                               nbins=15, color_discrete_map=COULEURS_STATUT,
                               labels={"resistance_compression": "Résistance (MPa)",
                                       "statut_conformite": "Statut"})
            fig.add_vline(x=seuil_res, line_dash="dash", line_color="red")
            fig.update_layout(title="Distribution des résistances en compression",
                              yaxis_title="Nombre d'essais", bargap=0.05)
            st.plotly_chart(mise_en_forme(fig), width="stretch")

        g3, g4 = st.columns(2)
        with g3:
            st.markdown("#### 🏗️ Performance par Type d'Ouvrage")
            perf = (df.groupby("type_ouvrage")
                      .agg(resistance_moyenne=("resistance_compression", "mean"),
                           taux_conformite=("statut_conformite",
                                            lambda s: (s == CONFORME).mean() * 100))
                      .reset_index())
            fig = px.bar(perf, x="type_ouvrage", y="resistance_moyenne",
                         color="taux_conformite", range_color=[0, 100],
                         color_continuous_scale=["red", "green"],
                         labels={"type_ouvrage": "Type d'ouvrage",
                                 "resistance_moyenne": "Résistance moyenne (MPa)",
                                 "taux_conformite": "Conformité (%)"})
            fig.update_layout(title="Performance par type d'ouvrage")
            st.plotly_chart(mise_en_forme(fig), width="stretch")
        with g4:
            st.markdown("#### 📋 Derniers Essais")
            der = df[["nom_ouvrage", "resistance_compression",
                      "statut_conformite", "date_essai"]].tail(5).copy()
            der["date_essai"] = der["date_essai"].dt.strftime("%d/%m/%Y")
            der.columns = ["Ouvrage", "Résistance (MPa)", "Statut", "Date essai"]
            st.dataframe(der, hide_index=True, width="stretch")
    else:
        st.info("Aucun essai enregistré. Ajoutez un essai dans le panneau de gauche.")

# ============================ Tab 2 : Analyse détaillée =====================
with tabs[1]:
    if len(df):
        a1, a2 = st.columns(2)
        with a1:
            st.markdown("#### 📊 Relation Résistance/Module")
            fig = px.scatter(df, x="resistance_compression", y="module_elasticite",
                             color="statut_conformite", hover_name="nom_ouvrage",
                             color_discrete_map=COULEURS_STATUT,
                             labels={"resistance_compression": "Résistance compression (MPa)",
                                     "module_elasticite": "Module d'élasticité (GPa)",
                                     "statut_conformite": "Statut"})
            fig.update_traces(marker=dict(size=10))
            if len(df) >= 2 and df["resistance_compression"].nunique() >= 2:
                pente, ordonnee = np.polyfit(df["resistance_compression"],
                                             df["module_elasticite"], 1)
                xs = np.array([df["resistance_compression"].min(),
                               df["resistance_compression"].max()])
                fig.add_trace(go.Scatter(x=xs, y=pente * xs + ordonnee, mode="lines",
                                         line=dict(color="blue", width=2),
                                         opacity=0.5, name="Régression linéaire"))
            fig.update_layout(title="Relation Résistance/Module d'élasticité")
            st.plotly_chart(mise_en_forme(fig), width="stretch")
        with a2:
            st.markdown("#### 💧 Affaissement vs Résistance")
            fig = px.scatter(df, x="affaissement", y="resistance_compression",
                             color="classe_resistance", hover_name="nom_ouvrage",
                             labels={"affaissement": "Affaissement (cm)",
                                     "resistance_compression": "Résistance compression (MPa)",
                                     "classe_resistance": "Classe"})
            fig.update_traces(marker=dict(size=10))
            fig.update_layout(title="Affaissement vs Résistance")
            st.plotly_chart(mise_en_forme(fig), width="stretch")

        a3, a4 = st.columns(2)
        with a3:
            st.markdown("#### 📈 Courbe Contrainte-Déformation")
            derniere = df.iloc[-1]
            fc = derniere["resistance_compression"]
            e_mpa = derniere["module_elasticite"] * 1000
            deformation = np.linspace(0, 0.0035, 100)
            contrainte = np.where(deformation <= fc / e_mpa, e_mpa * deformation, fc)
            fig = px.line(x=deformation, y=contrainte,
                          labels={"x": "Déformation", "y": "Contrainte (MPa)"})
            fig.update_traces(line=dict(color="#e74c3c", width=3))
            fig.update_layout(title=f"Courbe contrainte-déformation typique "
                                    f"({derniere['nom_ouvrage']})")
            st.plotly_chart(mise_en_forme(fig), width="stretch")
        with a4:
            st.markdown("#### 📋 Caractéristiques Mécaniques")
            tab = pd.DataFrame({
                "Caractéristique": ["Résistance moyenne", "Résistance caractéristique",
                                    "Module moyen", "Affaissement moyen",
                                    "Masse volumique moyenne", "Coefficient de variation"],
                "Valeur": [
                    fmt(df["resistance_compression"].mean(), 1, " MPa"),
                    fmt(calcul_resistance_caracteristique(df["resistance_compression"]), 1, " MPa"),
                    fmt(df["module_elasticite"].mean(), 1, " GPa"),
                    fmt(df["affaissement"].mean(), 1, " cm"),
                    fmt(df["masse_volumique"].mean(), 0, " kg/m³"),
                    fmt(coefficient_variation(df["resistance_compression"]), 1, "%"),
                ],
            })
            st.dataframe(tab, hide_index=True, width="stretch")
    else:
        st.info("Aucune donnée à analyser.")

# ============================ Tab 3 : Contrôle qualité ======================
with tabs[2]:
    if len(df):
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("#### 📝 Certificat de Conformité")
            taux_c = round((df["statut_conformite"] == CONFORME).mean() * 100, 1)
            fck = calcul_resistance_caracteristique(df["resistance_compression"])
            st.markdown("**CERTIFICAT DE CONFORMITÉ BÉTON**")
            st.divider()
            st.markdown(
                f"**Nombre d'essais :** {len(df)}  \n"
                f"**Résistance caractéristique :** {fck:.1f} MPa  \n"
                f"**Taux de conformité :** {taux_c}%  \n"
                f"**Module d'élasticité moyen :** {df['module_elasticite'].mean():.1f} GPa"
            )
            if taux_c >= 90 and fck >= seuil_res:
                st.success("✅ BÉTON CONFORME AUX EXIGENCES NORMATIVES")
            elif taux_c >= 80:
                st.warning("🟡 BÉTON ACCEPTABLE SOUS CONDITIONS")
            else:
                st.error("❌ BÉTON NON CONFORME - INVESTIGATION REQUISE")
        with q2:
            st.markdown("#### ⚠️ Points de Non-Conformité")
            nc = df[df["statut_conformite"] == NON_CONFORME]
            if len(nc):
                st.markdown("**POINTS DE NON-CONFORMITÉ**")
                st.divider()
                for _, ligne in nc.head(5).iterrows():
                    st.markdown(f"- {ligne['nom_ouvrage']} - Résistance : "
                                f"{ligne['resistance_compression']:.1f} MPa")
                st.markdown("**Actions correctives recommandées :**")
                st.markdown("- Vérifier le dosage en ciment\n"
                            "- Contrôler le rapport E/C\n"
                            "- Optimiser la cure du béton\n"
                            "- Revoir la formulation si nécessaire")
            else:
                st.success("✅ AUCUN POINT DE NON-CONFORMITÉ DÉTECTÉ")

        st.markdown("#### 📊 Cartes de Contrôle Statistique")
        d = df.copy()
        d["numero_essai"] = range(1, len(d) + 1)
        moyenne = d["resistance_compression"].mean()
        ecart = d["resistance_compression"].std(ddof=1) if len(d) > 1 else 0.0
        fig = px.scatter(d, x="numero_essai", y="resistance_compression",
                         color="statut_conformite", hover_name="nom_ouvrage",
                         color_discrete_map=COULEURS_STATUT,
                         labels={"numero_essai": "Numéro d'essai",
                                 "resistance_compression": "Résistance (MPa)",
                                 "statut_conformite": "Statut"})
        fig.update_traces(marker=dict(size=9))
        fig.add_trace(go.Scatter(x=d["numero_essai"], y=d["resistance_compression"],
                                 mode="lines", line=dict(color="#3498db", width=1),
                                 opacity=0.6, showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=moyenne, line_dash="dash", line_color="green",
                      annotation_text="Moyenne")
        fig.add_hline(y=seuil_res, line_dash="dash", line_color="red",
                      annotation_text="Seuil min.")
        fig.add_hline(y=moyenne + 2 * ecart, line_dash="dot", line_color="orange",
                      annotation_text="+2σ")
        fig.add_hline(y=moyenne - 2 * ecart, line_dash="dot", line_color="orange",
                      annotation_text="-2σ")
        fig.update_layout(title="Carte de contrôle des résistances")
        st.plotly_chart(mise_en_forme(fig, 430), width="stretch")
    else:
        st.info("Aucune donnée pour le contrôle qualité.")

# ============================ Tab 4 : Durabilité ============================
with tabs[3]:
    if len(df):
        ciment = st.radio("Type de ciment :", ["CEM II", "CEM I"], horizontal=True)
        res_moy_val = df["resistance_compression"].mean()
        mod_moy_val = df["module_elasticite"].mean()

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("#### 📈 Prédiction de Vieillissement")
            ages = [28, 90, 180, 365, 1825, 3650]
            etiquettes = ["28j", "3m", "6m", "1a", "5a", "10a"]
            pred = [prediction_resistance_long_terme(res_moy_val, a, ciment) for a in ages]
            fig = go.Figure(go.Scatter(x=ages, y=pred, mode="lines+markers",
                                       line=dict(color="#9b59b6", width=3),
                                       marker=dict(size=9)))
            fig.update_layout(title="Évolution prédite de la résistance dans le temps",
                              xaxis=dict(tickmode="array", tickvals=ages,
                                         ticktext=etiquettes, title="Âge"),
                              yaxis_title="Résistance prédite (MPa)")
            st.plotly_chart(mise_en_forme(fig), width="stretch")
        with d2:
            st.markdown("#### 🛡️ Indices de Durabilité")
            tab = pd.DataFrame({
                "Indice": ["Résistance à 1 an", "Résistance à 5 ans",
                           "Résistance à 10 ans", "Module à long terme",
                           "Indice de durabilité"],
                "Valeur": [
                    fmt(prediction_resistance_long_terme(res_moy_val, 365, ciment), 1, " MPa"),
                    fmt(prediction_resistance_long_terme(res_moy_val, 1825, ciment), 1, " MPa"),
                    fmt(prediction_resistance_long_terme(res_moy_val, 3650, ciment), 1, " MPa"),
                    fmt(mod_moy_val * 1.1, 1, " GPa"),
                    "Élevé" if res_moy_val > 30 else "Moyen",
                ],
            })
            st.dataframe(tab, hide_index=True, width="stretch")

        st.markdown("#### 📋 Recommandations Durabilité")
        st.markdown("**RECOMMANDATIONS POUR LA DURABILITÉ**")
        st.divider()
        if res_moy_val < 30:
            st.markdown("- ✅ Classe d'exposition XC1-XC2 acceptable\n"
                        "- ⚠️ Éviter les environnements agressifs\n"
                        "- 🔍 Surveillance renforcée recommandée")
        elif res_moy_val < 40:
            st.markdown("- ✅ Convient pour environnements modérés (XC3-XC4)\n"
                        "- 🛡️ Protection contre les chlorures conseillée\n"
                        "- 💧 Maîtrise de la perméabilité nécessaire")
        else:
            st.markdown("- ✅ Excellente durabilité (XD, XS)\n"
                        "- 🏗️ Adapté aux environnements marins\n"
                        "- ⏳ Longue durée de vie garantie")
        st.markdown("**Mesures préventives :**")
        st.markdown("- Cure humide pendant 7 jours minimum\n"
                    "- Enrobage des armatures ≥ 4 cm\n"
                    "- Utilisation de produits de cure\n"
                    "- Contrôle de la fissuration")
    else:
        st.info("Aucune donnée pour l'analyse de durabilité.")

# ============================ Tab 5 : Base de données =======================
with tabs[4]:
    st.markdown("#### Archive des essais sur béton")
    aff = df.copy()
    for col in ("date_fabrication", "date_essai"):
        aff[col] = aff[col].dt.strftime("%d/%m/%Y")
    st.dataframe(aff, hide_index=True, width="stretch")

    # Export rapport synthétique
    if len(df):
        rapport = pd.DataFrame({
            "Paramètre": ["Nombre total d'essais", "Résistance moyenne (MPa)",
                          "Résistance caractéristique (MPa)",
                          "Module d'élasticité moyen (GPa)", "Affaissement moyen (cm)",
                          "Taux de conformité (%)", "Essais non conformes",
                          "Coefficient de variation (%)"],
            "Valeur": [
                len(df),
                round(df["resistance_compression"].mean(), 1),
                round(calcul_resistance_caracteristique(df["resistance_compression"]), 1),
                round(df["module_elasticite"].mean(), 1),
                round(df["affaissement"].mean(), 1),
                round((df["statut_conformite"] == CONFORME).mean() * 100, 1),
                int((df["statut_conformite"] == NON_CONFORME).sum()),
                round(coefficient_variation(df["resistance_compression"]), 1),
            ],
        })
    else:
        rapport = pd.DataFrame({"Paramètre": [], "Valeur": []})

    def en_csv(frame):
        # Équivalent de write.csv2 : séparateur ";" et virgule décimale
        return frame.to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig")

    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        st.download_button("📥 Exporter Rapport", data=en_csv(rapport),
                           file_name=f"rapport_beton_{pd.Timestamp.today().date()}.csv",
                           mime="text/csv", width="stretch")
    with b2:
        st.download_button("📥 Exporter Excel (CSV)", data=en_csv(aff),
                           file_name=f"analyse_beton_{pd.Timestamp.today().date()}.csv",
                           mime="text/csv", width="stretch")

# ============================ Tab 6 : Guide technique =======================
with tabs[5]:
    st.markdown("#### 🏗️ Normes et Références Béton")
    st.markdown("""
**Classes de résistance (NF EN 206/CN) :**
- **C16/20 :** Béton de propreté
- **C20/25 :** Fondations, dallages
- **C25/30 :** Ouvrages courants
- **C30/37 :** Ouvrages spéciaux
- **C35/45 :** Ouvrages précontraints

**Critères de conformité (NF EN 206-1) :**
- Résistance moyenne ≥ fck + λ × σ
- Aucun essai < fck - 4 MPa
- λ = 1.48 pour n ≥ 15 essais

**Relations empiriques :**
- **Résistance traction :** fctm ≈ 0.3 × fck^(2/3)
- **Module d'élasticité :** Ecm ≈ 22 × (fcm/10)^0.3
- **Résistance flexion :** fctm,fl ≈ max(1.6 - h/1000 ; 1) × fctm

**Affaissements recommandés :**
- Fondations : 5-10 cm
- Poteaux : 8-12 cm
- Dalles : 10-15 cm
- Voiles : 12-18 cm

**Évolution résistance dans le temps :**
- fcm(t) = fcm × exp[s × (1 - √(28/t))]
- s = 0.25 (ciment CEM I)
- s = 0.20 (ciment CEM II)
""")
