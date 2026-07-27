
TEST_CASES = [
    # 1. FACTUAL QUERY (Retrieval precision, Answer accuracy, Citation faithfulness)
    {
        "query": "What is Qatar's GDP according to the report?",
        "relevant_pages": [4, 5],
        "reference_answer": "The report states that Qatar’s nominal GDP is estimated at around USD 220–230 billion in recent years, reflecting moderate growth supported by the hydrocarbon sector and a gradual recovery in non-hydrocarbon activity."
    },
    {
        "query": "How does the report characterize Qatar’s overall economic performance in recent years?",
        "relevant_pages": [4, 5, 11],
        "reference_answer": "The report characterizes Qatar’s economic performance as resilient, with positive growth, strong buffers, and stable macroeconomic conditions."
    },
    {
        "query": "What does the report say about inflation trends in Qatar?",
        "relevant_pages": [4, 5],
        "reference_answer": "Inflation is described as moderate, reflecting easing price pressures and stable domestic conditions."
    },

    # 2. DATA / TABLE QUERY (Table detection, Numeric extraction, Multimodal grounding)
    {
        "query": "What does the report indicate about Qatar’s recent GDP growth performance?",
        "relevant_pages": [4, 5],
        "reference_answer": "According to the macroeconomic tables and discussion, GDP growth moderated after earlier expansion and remained positive, with growth driven mainly by the non-hydrocarbon sector while hydrocarbon output was broadly stable."
    },
    {
        "query": "What fiscal information is presented in the central government operations table?",
        "relevant_pages": [36],
        "reference_answer": "The table presents data on government revenues, expenditures, fiscal balances, and financing."
    },
    {
        "query": "What does the balance of payments table reveal about Qatar’s external position?",
        "relevant_pages": [37],
        "reference_answer": "The table shows a strong external position supported by exports and favorable current account balances."
    },

    # 3. BROAD SYNTHESIS QUERY (Recall, Coverage, Multi-section reasoning)
    {
        "query": "What are the main economic sectors discussed in the report?",
        "relevant_pages": [1, 2, 3],
        "reference_answer": "The report highlights hydrocarbons as the dominant sector, alongside growing contributions from non-hydrocarbon activities such as construction, trade, transport, financial services, and other services linked to diversification efforts."
    },
    {
        "query": "What are the key risks to Qatar’s economic outlook identified in the report?",
        "relevant_pages": [9, 10],
        "reference_answer": "Key risks include global economic uncertainty, volatility in energy prices, and challenges related to reform implementation."
    },
    {
        "query": "What policy priorities does the report emphasize for the medium term?",
        "relevant_pages": [32, 33, 34],
        "reference_answer": "The report emphasizes structural reforms, diversification, fiscal sustainability, and private sector–led growth as key medium-term priorities."
    },

    # 4. CHART / FIGURE QUERY (Chart detection, OCR accuracy, Cross-modal reasoning)
    {
        "query": "What does the non-hydrocarbon GDP growth chart illustrate?",
        "relevant_pages": [49, 50],
        "reference_answer": "The chart illustrates a slowdown in non-hydrocarbon GDP growth following a period of strong expansion, followed by a gradual recovery toward more sustainable growth rates."
    },
    {
        "query": "What trend is shown in the fiscal risk assessment figure?",
        "relevant_pages": [54],
        "reference_answer": "The figure indicates that fiscal risks are assessed as contained and manageable over the medium term."
    },
    {
        "query": "What visual evidence supports the assessment of economic normalization?",
        "relevant_pages": [7, 49],
        "reference_answer": "Figures show a transition from earlier expansion toward normalization, with growth stabilizing at more moderate levels."
    },

    # 5. NEGATIVE / FAITHFULNESS QUERY (Hallucination prevention, Abstention correctness)
    {
        "query": "Does the document provide information on cryptocurrency regulation in Qatar?",
        "relevant_pages": [],
        "reference_answer": "No. The document does not contain any discussion or analysis of cryptocurrency regulation in Qatar."
    },
    {
        "query": "Does the report include data on individual household income distribution?",
        "relevant_pages": [],
        "reference_answer": "No. The report does not include information on individual household income distribution."
    },
    {
        "query": "Is there any discussion of social media policy or digital platform regulation?",
        "relevant_pages": [],
        "reference_answer": "No. The document does not discuss social media policy or digital platform regulation."
    }
]
