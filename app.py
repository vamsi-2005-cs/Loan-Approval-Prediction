import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a8a 55%,
            #2563eb 100%
        );
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    .approved-card {
        padding: 1.6rem;
        border-radius: 18px;
        text-align: center;
        background: #ecfdf5;
        border: 2px solid #10b981;
        color: #065f46;
    }

    .declined-card {
        padding: 1.6rem;
        border-radius: 18px;
        text-align: center;
        background: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area"
]

REQUIRED = [
    "Loan_ID"
] + FEATURES + [
    "Loan_Status"
]


# ============================================================
# ENCODING MAPPINGS
# ============================================================

GENDER = {
    "Female": 0,
    "Male": 1
}

YES_NO = {
    "No": 0,
    "Yes": 1
}

DEPENDENTS = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}

EDUCATION = {
    "Not Graduate": 0,
    "Graduate": 1
}

PROPERTY = {
    "Semiurban": 0,
    "Urban": 1,
    "Rural": 2
}


# ============================================================
# REFERENCE NOTEBOOK RESULTS
# ============================================================

REFERENCE_SCORES = {
    "Gradient Boosting": 77.69,
    "Random Forest": 75.90,
    "Decision Tree": 70.04,
    "K-Nearest Neighbor": 61.40,
    "SVM": 80.78,
}


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(df):

    d = df.copy()

    categorical_mode = [
        "Gender",
        "Married",
        "Dependents",
        "Self_Employed",
        "Loan_Amount_Term",
        "Credit_History"
    ]

    # Fill categorical missing values with mode
    for col in categorical_mode:

        if d[col].isnull().any():

            mode_value = d[col].mode()

            if not mode_value.empty:

                d[col] = d[col].fillna(
                    mode_value.iloc[0]
                )

    # Fill LoanAmount missing values with median
    if d["LoanAmount"].isnull().any():

        median_value = d["LoanAmount"].median()

        d["LoanAmount"] = d["LoanAmount"].fillna(
            median_value
        )

    # Apply categorical mappings
    d["Gender"] = d["Gender"].replace(
        GENDER
    )

    d["Married"] = d["Married"].replace(
        YES_NO
    )

    d["Dependents"] = d["Dependents"].replace(
        DEPENDENTS
    )

    d["Education"] = d["Education"].replace(
        EDUCATION
    )

    d["Self_Employed"] = d["Self_Employed"].replace(
        YES_NO
    )

    d["Property_Area"] = d["Property_Area"].replace(
        PROPERTY
    )

    # Convert model columns to numeric
    for col in FEATURES:

        d[col] = pd.to_numeric(
            d[col],
            errors="coerce"
        )

    # Remove rows that still contain invalid values
    d = d.dropna(
        subset=FEATURES
    )

    return d


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(csv_path):

    raw = pd.read_csv(csv_path)

    # Check required columns
    missing = [
        col
        for col in REQUIRED
        if col not in raw.columns
    ]

    if missing:

        raise ValueError(
            f"Dataset is missing columns: {missing}"
        )

    d = preprocess(raw)

    X = d[FEATURES]

    # Match target rows after preprocessing
    y = raw.loc[
        d.index,
        "Loan_Status"
    ]

    models = {

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=10,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "K-Nearest Neighbor":
            KNeighborsClassifier(),

        "SVM":
            svm.LinearSVC(
                max_iter=5000,
                random_state=42
            )
    }

    scores = {}

    for name, model in models.items():

        score = cross_val_score(
            model,
            X,
            y,
            cv=5
        ).mean()

        scores[name] = float(score)

    # Find best model
    best_name = max(
        scores,
        key=scores.get
    )

    best_model = models[
        best_name
    ]

    # Train best model using complete data
    best_model.fit(
        X,
        y
    )

    return (
        raw,
        scores,
        best_name,
        best_model
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🏦 Loan Approval Predictor"
)

st.caption(
    "Predict loan eligibility using machine learning "
    "and explore model performance interactively."
)


# ============================================================
# DATASET SELECTION
# ============================================================

default_csv = (
    Path(__file__).parent
    / "data"
    / "train.csv"
)

csv_path = None


# Automatically load local train.csv
if default_csv.exists():

    csv_path = str(
        default_csv
    )


# Optional uploader
with st.expander(
    "📁 Change Training Dataset",
    expanded=False
):

    st.write(
        "The application automatically uses "
        "`data/train.csv` when available."
    )

    uploaded = st.file_uploader(
        "Upload another training CSV",
        type=["csv"],
        help=(
            "The CSV must contain the same columns "
            "used by the training notebook."
        )
    )

    if uploaded is not None:

        upload_directory = (
            Path(__file__).parent
            / "data"
        )

        upload_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_path = (
            upload_directory
            / "_uploaded_train.csv"
        )

        temp_path.write_bytes(
            uploaded.getvalue()
        )

        csv_path = str(
            temp_path
        )

        st.success(
            "Custom training dataset loaded."
        )


# ============================================================
# TRAINING
# ============================================================

raw = None
scores = None
best_name = None
best_model = None


if csv_path:

    try:

        with st.spinner(
            "Training machine-learning models..."
        ):

            (
                raw,
                scores,
                best_name,
                best_model
            ) = train_models(
                csv_path
            )

    except Exception as error:

        st.error(
            f"Could not train the model: {error}"
        )

else:

    st.warning(
        "No training dataset was found. "
        "Place train.csv inside the data folder "
        "or upload a CSV above."
    )


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

if raw is not None:

    total_rows = len(raw)

    approved_count = int(
        (
            raw["Loan_Status"] == "Y"
        ).sum()
    )

    declined_count = int(
        (
            raw["Loan_Status"] == "N"
        ).sum()
    )

    approval_rate = (
        approved_count
        / total_rows
        * 100
        if total_rows > 0
        else 0
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "📄 Training Records",
        f"{total_rows:,}"
    )

    c2.metric(
        "🧩 Features",
        len(FEATURES)
    )

    c3.metric(
        "🤖 Best Model",
        best_name
    )

    c4.metric(
        "🎯 CV Accuracy",
        f"{scores[best_name] * 100:.2f}%"
    )


# ============================================================
# TABS
# ============================================================

tab_predict, tab_models, tab_data, tab_about = st.tabs(
    [
        "🔮 Predict Loan",
        "📊 Model Performance",
        "🗃️ Dataset Explorer",
        "ℹ️ About"
    ]
)


# ============================================================
# PREDICTION TAB
# ============================================================

with tab_predict:

    st.subheader(
        "🔮 Applicant Assessment"
    )

    st.write(
        "Enter the applicant information below "
        "to generate a loan approval prediction."
    )

    if best_model is None:

        st.info(
            "Load a training dataset to enable predictions."
        )

    else:

        # ----------------------------------------------------
        # RESET VERSION
        # ----------------------------------------------------

        if "reset_version" not in st.session_state:

            st.session_state.reset_version = 0


        reset_version = st.session_state.reset_version


        # ----------------------------------------------------
        # PREDICTION FORM
        # ----------------------------------------------------

        with st.form(
            "loan_prediction_form"
        ):

            # ------------------------------------------------
            # PERSONAL INFORMATION
            # ------------------------------------------------

            st.markdown(
                "### 👤 Personal Information"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                gender = st.selectbox(
                    "Gender",
                    [
                        "Male",
                        "Female"
                    ],
                    index=0,
                    key=f"gender_{reset_version}"
                )

            with col2:

                married = st.selectbox(
                    "Married",
                    [
                        "Yes",
                        "No"
                    ],
                    index=0,
                    key=f"married_{reset_version}"
                )

            with col3:

                dependents = st.selectbox(
                    "Dependents",
                    [
                        "0",
                        "1",
                        "2",
                        "3+"
                    ],
                    index=0,
                    key=f"dependents_{reset_version}"
                )


            # ------------------------------------------------
            # EDUCATION & EMPLOYMENT
            # ------------------------------------------------

            st.markdown(
                "### 🎓 Education & Employment"
            )

            col4, col5, col6 = st.columns(3)

            with col4:

                education = st.selectbox(
                    "Education",
                    [
                        "Graduate",
                        "Not Graduate"
                    ],
                    index=0,
                    key=f"education_{reset_version}"
                )

            with col5:

                self_employed = st.selectbox(
                    "Self Employed",
                    [
                        "No",
                        "Yes"
                    ],
                    index=0,
                    key=f"self_employed_{reset_version}"
                )

            with col6:

                credit_history = st.selectbox(
                    "Credit History",
                    [
                        1,
                        0
                    ],
                    index=0,
                    key=f"credit_history_{reset_version}",
                    format_func=lambda value:
                        (
                            "Good Credit History"
                            if value == 1
                            else
                            "Poor Credit History"
                        )
                )


            # ------------------------------------------------
            # FINANCIAL INFORMATION
            # ------------------------------------------------

            st.markdown(
                "### 💰 Financial Information"
            )

            col7, col8, col9 = st.columns(3)

            with col7:

                applicant_income = st.number_input(
                    "Applicant Income",
                    min_value=0.0,
                    value=5000.0,
                    step=500.0,
                    key=f"applicant_income_{reset_version}",
                    help="Applicant's income."
                )

            with col8:

                coapplicant_income = st.number_input(
                    "Coapplicant Income",
                    min_value=0.0,
                    value=1500.0,
                    step=500.0,
                    key=f"coapplicant_income_{reset_version}",
                    help="Coapplicant's income."
                )

            with col9:

                loan_amount = st.number_input(
                    "Loan Amount (thousands)",
                    min_value=0.0,
                    value=150.0,
                    step=10.0,
                    key=f"loan_amount_{reset_version}",
                    help=(
                        "Loan amount in thousands, "
                        "matching the training dataset."
                    )
                )


            # ------------------------------------------------
            # LOAN DETAILS
            # ------------------------------------------------

            st.markdown(
                "### 🏠 Loan Details"
            )

            col10, col11 = st.columns(2)

            with col10:

                loan_term = st.selectbox(
                    "Loan Amount Term",
                    [
                        60,
                        120,
                        180,
                        240,
                        300,
                        360,
                        480
                    ],
                    index=5,
                    key=f"loan_term_{reset_version}",
                    format_func=lambda value:
                        f"{value} months"
                )

            with col11:

                property_area = st.selectbox(
                    "Property Area",
                    [
                        "Urban",
                        "Semiurban",
                        "Rural"
                    ],
                    index=0,
                    key=f"property_area_{reset_version}"
                )


            st.write("")


            # ------------------------------------------------
            # BUTTONS
            # ------------------------------------------------

            button1, button2 = st.columns(
                [3, 1]
            )

            with button1:

                submitted = st.form_submit_button(
                    "🚀 Predict Loan Status",
                    use_container_width=True,
                    type="primary"
                )

            with button2:

                reset = st.form_submit_button(
                    "🔄 Reset Input",
                    use_container_width=True
                )


        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        if reset:

            # Increment the widget key version.
            # This creates completely new widgets
            # using their original default values.
            st.session_state.reset_version += 1

            # Clear any previous prediction
            if "prediction_result" in st.session_state:

                del st.session_state[
                    "prediction_result"
                ]

            st.rerun()


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if submitted:

            applicant = pd.DataFrame(
                [{
                    "Gender":
                        GENDER[gender],

                    "Married":
                        YES_NO[married],

                    "Dependents":
                        DEPENDENTS[dependents],

                    "Education":
                        EDUCATION[education],

                    "Self_Employed":
                        YES_NO[self_employed],

                    "ApplicantIncome":
                        applicant_income,

                    "CoapplicantIncome":
                        coapplicant_income,

                    "LoanAmount":
                        loan_amount,

                    "Loan_Amount_Term":
                        loan_term,

                    "Credit_History":
                        credit_history,

                    "Property_Area":
                        PROPERTY[property_area]
                }],
                columns=FEATURES
            )


            prediction = best_model.predict(
                applicant
            )[0]


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = None

            if hasattr(
                best_model,
                "predict_proba"
            ):

                probabilities = (
                    best_model
                    .predict_proba(
                        applicant
                    )[0]
                )

                classes = list(
                    best_model.classes_
                )

                if prediction in classes:

                    index = classes.index(
                        prediction
                    )

                    confidence = (
                        probabilities[index]
                        * 100
                    )

            elif hasattr(
                best_model,
                "decision_function"
            ):

                decision = float(
                    best_model
                    .decision_function(
                        applicant
                    )[0]
                )

                # Convert decision score into
                # a simple visual confidence value.
                confidence = (
                    100 /
                    (
                        1 +
                        np.exp(
                            -abs(decision)
                        )
                    )
                )


            # ------------------------------------------------
            # SAVE RESULT IN SESSION STATE
            # ------------------------------------------------

            st.session_state[
                "prediction_result"
            ] = {
                "prediction": prediction,
                "confidence": confidence,
                "applicant": applicant,
                "gender": gender,
                "married": married,
                "dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "credit_history": credit_history,
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "property_area": property_area
            }


        # ----------------------------------------------------
        # SHOW SAVED RESULT
        # ----------------------------------------------------

        if (
            "prediction_result"
            in st.session_state
        ):

            result = st.session_state[
                "prediction_result"
            ]

            prediction = result[
                "prediction"
            ]

            confidence = result[
                "confidence"
            ]

            applicant = result[
                "applicant"
            ]


            st.write("")
            st.divider()


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction == "Y":

                st.success(
                    "## ✅ LOAN APPROVED\n\n"
                    "The trained model predicts "
                    "that this applicant is eligible "
                    "for loan approval."
                )

            else:

                st.error(
                    "## ❌ LOAN DECLINED\n\n"
                    "The trained model predicts "
                    "that this applicant is not "
                    "eligible for loan approval."
                )


            st.write("")


            result1, result2, result3 = st.columns(3)

            with result1:

                st.metric(
                    "🤖 Selected Model",
                    best_name
                )

            with result2:

                st.metric(
                    "🎯 CV Accuracy",
                    f"{scores[best_name] * 100:.2f}%"
                )

            with result3:

                if confidence is not None:

                    st.metric(
                        "📈 Prediction Confidence",
                        f"{confidence:.1f}%"
                    )

                else:

                    st.metric(
                        "📈 Confidence",
                        "N/A"
                    )


            if confidence is not None:

                st.write(
                    "Prediction confidence"
                )

                st.progress(
                    min(
                        int(confidence),
                        100
                    )
                )


            # ------------------------------------------------
            # APPLICANT SUMMARY
            # ------------------------------------------------

            with st.expander(
                "👁️ View Applicant Information"
            ):

                summary = pd.DataFrame(
                    {
                        "Field": [
                            "Gender",
                            "Married",
                            "Dependents",
                            "Education",
                            "Self Employed",
                            "Applicant Income",
                            "Coapplicant Income",
                            "Loan Amount",
                            "Loan Term",
                            "Credit History",
                            "Property Area"
                        ],

                        "Value": [
                            result["gender"],
                            result["married"],
                            result["dependents"],
                            result["education"],
                            result["self_employed"],
                            f"{result['applicant_income']:,.0f}",
                            f"{result['coapplicant_income']:,.0f}",
                            f"{result['loan_amount']:,.0f}",
                            f"{result['loan_term']} months",
                            (
                                "Good"
                                if result["credit_history"] == 1
                                else "Poor"
                            ),
                            result["property_area"]
                        ]
                    }
                )

                st.dataframe(
                    summary,
                    hide_index=True,
                    use_container_width=True
                )


            # ------------------------------------------------
            # ENCODED INPUT
            # ------------------------------------------------

            with st.expander(
                "🔢 View Encoded Input"
            ):

                st.dataframe(
                    applicant,
                    hide_index=True,
                    use_container_width=True
                )


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with tab_models:

    st.subheader(
        "📊 Model Performance"
    )

    st.write(
        "The five classifiers are compared using "
        "5-fold cross-validation."
    )

    if scores:

        model_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Accuracy": score * 100
                }

                for name, score
                in scores.items()
            ]
        )

        model_df = model_df.sort_values(
            "Accuracy",
            ascending=False
        )


        # Best model
        st.success(
            f"🏆 Best model: **{best_name}** "
            f"with **{scores[best_name] * 100:.2f}%** "
            f"5-fold CV accuracy."
        )


        st.markdown(
            "### 📈 Current Model Comparison"
        )

        chart_data = model_df.set_index(
            "Model"
        )

        st.bar_chart(
            chart_data
        )


        st.dataframe(
            model_df.style.format(
                {
                    "Accuracy": "{:.2f}%"
                }
            ),
            hide_index=True,
            use_container_width=True
        )


        # ----------------------------------------------------
        # REFERENCE RESULTS
        # ----------------------------------------------------

        st.markdown(
            "### 📚 Notebook Reference Results"
        )

        reference_df = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Notebook Accuracy": score
                }

                for name, score
                in REFERENCE_SCORES.items()
            ]
        )

        reference_df = reference_df.sort_values(
            "Notebook Accuracy",
            ascending=False
        )


        st.dataframe(
            reference_df.style.format(
                {
                    "Notebook Accuracy": "{:.2f}%"
                }
            ),
            hide_index=True,
            use_container_width=True
        )


        st.bar_chart(
            reference_df.set_index(
                "Model"
            )[
                "Notebook Accuracy"
            ]
        )


    else:

        st.info(
            "Load a training dataset to calculate "
            "model performance."
        )


# ============================================================
# DATASET EXPLORER TAB
# ============================================================

with tab_data:

    st.subheader(
        "🗃️ Interactive Dataset Explorer"
    )

    if raw is not None:

        # ----------------------------------------------------
        # DATASET METRICS
        # ----------------------------------------------------

        total_records = len(raw)

        total_columns = len(
            raw.columns
        )

        approved = int(
            (
                raw["Loan_Status"] == "Y"
            ).sum()
        )

        declined = int(
            (
                raw["Loan_Status"] == "N"
            ).sum()
        )


        a, b, c, d = st.columns(4)

        a.metric(
            "Total Records",
            f"{total_records:,}"
        )

        b.metric(
            "Columns",
            total_columns
        )

        c.metric(
            "Approved",
            approved
        )

        d.metric(
            "Declined",
            declined
        )


        st.divider()


        # ----------------------------------------------------
        # FILTERS
        # ----------------------------------------------------

        st.markdown(
            "### 🔎 Filter Dataset"
        )

        filter1, filter2 = st.columns(2)

        with filter1:

            search = st.text_input(
                "Search Loan ID",
                placeholder="e.g. LP001002"
            )

        with filter2:

            available_statuses = sorted(
                raw["Loan_Status"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_status = st.multiselect(
                "Loan Status",
                available_statuses,
                default=available_statuses,
                format_func=lambda value:
                    (
                        "Approved (Y)"
                        if value == "Y"
                        else "Declined (N)"
                    )
            )


        filtered = raw.copy()


        if search:

            filtered = filtered[
                filtered["Loan_ID"]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]


        if selected_status:

            filtered = filtered[
                filtered["Loan_Status"].isin(
                    selected_status
                )
            ]

        else:

            filtered = filtered.iloc[0:0]


        st.caption(
            f"Showing {len(filtered):,} record(s)"
        )


        st.dataframe(
            filtered,
            use_container_width=True,
            height=450
        )


        # ----------------------------------------------------
        # DATA VISUALIZATION
        # ----------------------------------------------------

        st.markdown(
            "### 📈 Dataset Insights"
        )

        chart1, chart2 = st.columns(2)


        with chart1:

            st.markdown(
                "**Loan Status Distribution**"
            )

            status_counts = (
                raw["Loan_Status"]
                .value_counts()
                .rename(
                    index={
                        "Y": "Approved",
                        "N": "Declined"
                    }
                )
            )

            st.bar_chart(
                status_counts
            )


        with chart2:

            st.markdown(
                "**Education Distribution**"
            )

            education_counts = (
                raw["Education"]
                .value_counts()
            )

            st.bar_chart(
                education_counts
            )


        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        with st.expander(
            "🔧 View Model Features"
        ):

            for feature in FEATURES:

                st.write(
                    f"- `{feature}`"
                )

    else:

        st.info(
            "Load a training dataset to explore the data."
        )


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.subheader(
        "ℹ️ About This Project"
    )

    st.markdown(
        """
        ### 🏦 Loan Approval Prediction

        This application provides an interactive interface
        for the Loan Approval Prediction machine-learning
        project.

        ### 🔄 Machine Learning Workflow

        1. Load the training dataset.
        2. Handle missing categorical values using the mode.
        3. Fill missing `LoanAmount` values using the median.
        4. Encode categorical variables.
        5. Evaluate five machine-learning classifiers.
        6. Use 5-fold cross-validation.
        7. Select the highest-performing classifier.
        8. Train the selected classifier on the complete dataset.
        9. Predict `Loan_Status`.

        ### 🤖 Models

        - Gradient Boosting
        - Random Forest
        - Decision Tree
        - K-Nearest Neighbor
        - Support Vector Machine

        ### 🎯 Prediction Output

        The model predicts one of two outcomes:

        - **Y** → Loan Approved
        - **N** → Loan Declined

        ### ⚠️ Important

        This application is intended for educational
        and demonstration purposes. It should not be used
        as the sole basis for real-world lending decisions.
        """
    )

    st.info(
        "According to the original notebook results, "
        "SVM achieved 80.78% 5-fold CV accuracy. "
        "The application independently recalculates model "
        "performance when the training dataset is loaded."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏦 Loan Approval Predictor • "
    "Machine Learning Demonstration"
)
