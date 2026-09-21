"""Test cases for RAG evaluation"""

# Test cases for Qatar Economic Report
TEST_CASES = [
    {
        'query': "What is Qatar's GDP?",
        'relevant_pages': [4, 5, 42],
        'reference_answer': "Qatar's GDP is approximately $235 billion in 2023, showing 3.2% growth from the previous year."
    },
    {
        'query': "What is the inflation rate?",
        'relevant_pages': [4, 5],
        'reference_answer': "The inflation rate in Qatar is around 3.2% in 2023."
    },
    {
        'query': "What are the main economic sectors?",
        'relevant_pages': [1, 2, 3],
        'reference_answer': "Qatar's main economic sectors include oil and gas (dominant), tourism, financial services, and construction."
    },
    {
        'query': "What is the fiscal deficit?",
        'relevant_pages': [41, 42],
        'reference_answer': "The fiscal deficit has narrowed in recent years due to improved oil revenues and fiscal consolidation measures."
    },
    {
        'query': "What are the economic growth drivers?",
        'relevant_pages': [1, 2, 3, 4],
        'reference_answer': "Economic growth is driven by oil and gas exports, tourism expansion, infrastructure development, and economic diversification efforts."
    },
    {
        'query': "Show me GDP growth data",
        'relevant_pages': [4, 5, 42],
        'reference_answer': "GDP growth has been positive, with rates around 3-4% annually in recent years."
    },
    {
        'query': "What are the economic indicators?",
        'relevant_pages': [4, 5],
        'reference_answer': "Key economic indicators include GDP growth of 3.2%, inflation at 3.2%, and strong fiscal position."
    },
    {
        'query': "What is the unemployment rate?",
        'relevant_pages': [4, 5],
        'reference_answer': "The unemployment rate in Qatar remains low, reflecting strong labor market conditions."
    },
    {
        "query": "What exchange rate arrangement does Qatar follow?",
        "relevant_pages": [2, 3],
        "reference_answer": "Qatar follows a conventional pegged exchange rate arrangement, with the Qatari riyal pegged to the U.S. dollar."
    },
    {
        "query": "When were the discussions for the latest Article IV consultation held?",
        "relevant_pages": [3],
        "reference_answer": "The Article IV consultation discussions were held in November 2024."
    },
    {
        "query": "What information is presented in the table of selected economic indicators?",
        "relevant_pages": [35],
        "reference_answer": "The table presents key macroeconomic indicators including growth, inflation, fiscal balances, and external sector metrics."
    },
    {
        "query": "According to the central government operations table, what type of fiscal information is reported?",
        "relevant_pages": [36],
        "reference_answer": "The table reports central government revenues, expenditures, balances, and financing details."
    },
    {
        "query": "What does the balance of payments table summarize?",
        "relevant_pages": [37],
        "reference_answer": "The table summarizes exports, imports, current account balance, and other external transactions."
    },
    {
        "query": "What key data is included in the monetary survey table?",
        "relevant_pages": [38],
        "reference_answer": "The monetary survey table includes money supply, credit aggregates, and financial sector indicators."
    },
    {
        "query": "What does the table on sovereign credit ratings indicate about Qatar?",
        "relevant_pages": [41],
        "reference_answer": "The table indicates that Qatar maintains strong sovereign credit ratings from major international agencies."
    },
    {
        "query": "What trend is illustrated in the economic diversification indicators figure?",
        "relevant_pages": [7, 8],
        "reference_answer": "The figure illustrates progress in some diversification indicators while highlighting remaining structural gaps."
    },
    {
        "query": "What does the non-hydrocarbon GDP growth nowcasting chart show?",
        "relevant_pages": [49, 50],
        "reference_answer": "The chart shows a slowdown in non-hydrocarbon growth followed by a gradual recovery."
    },
    {
        "query": "What assessment is conveyed by the fiscal risk assessment figure?",
        "relevant_pages": [54, 59],
        "reference_answer": "The figure conveys that fiscal risks are assessed as low to moderate over the medium term."
    },
    {
        "query": "What does the chart indicate when comparing non-hydrocarbon growth trends over time?",
        "relevant_pages": [49, 50],
        "reference_answer": "It indicates variation over time with a period of slowdown and subsequent recovery."
    },
    {
        "query": "What visual evidence supports the report’s assessment of economic stability?",
        "relevant_pages": [35, 49, 54],
        "reference_answer": "Tables and figures together visually support an assessment of stable conditions and manageable risks."
    },
    {
        "query": "What text information is extracted from figure captions related to fiscal risk?",
        "relevant_pages": [54],
        "reference_answer": "The captions describe the methodology and conclusion that fiscal risks are contained."
    },
    {
        "query": "What information can be read from the chart axes in the non-hydrocarbon GDP growth figure?",
        "relevant_pages": [49],
        "reference_answer": "The axes show growth rates over time for non-hydrocarbon GDP."
    },
    {
        "query": "What does the document’s visual content indicate about post-normalization economic trends?",
        "relevant_pages": [7, 49],
        "reference_answer": "The visual content indicates normalization after earlier expansion with gradual recovery."
    },
    {
        "query": "What tabular evidence supports the assessment of external sector strength?",
        "relevant_pages": [37, 41],
        "reference_answer": "External sector tables show strong balances and indicators supporting external strength."
    },
    {
        "query": "Is any numerical information visible only in charts and not explicitly repeated in the text?",
        "relevant_pages": [49, 54],
        "reference_answer": "Some trends are visually represented in charts and summarized rather than numerically detailed in the text."
    }
,

    {
        "query": "When did the IMF Executive Board conclude the Article IV consultation for Qatar?",
        "relevant_pages": [11],
        "reference_answer": "The IMF Executive Board concluded the Article IV consultation in January 2025."
    },
    {
        "query": "What is the inflation outlook mentioned in the report?",
        "relevant_pages": [12],
        "reference_answer": "Headline inflation is expected to ease in the near term and converge to around 2 percent over the medium term."
    },
    {
        "query": "How is the condition of Qatar’s banking sector described in the report?",
        "relevant_pages": [12],
        "reference_answer": "The banking sector is described as well-capitalized, liquid, and profitable."
    },
    {
        "query": "What are the main objectives of Qatar’s Third National Development Strategy?",
        "relevant_pages": [42, 43, 44],
        "reference_answer": "The strategy aims to promote economic diversification, private sector-led growth, human capital development, government efficiency, and sustainability."
    },
    {
        "query": "What does the table on sovereign credit ratings indicate?",
        "relevant_pages": [41],
        "reference_answer": "The table shows that Qatar maintains strong sovereign credit ratings from major international rating agencies."
    },
    {
        "query": "What trend is shown in the figure on non-hydrocarbon GDP growth?",
        "relevant_pages": [49, 50],
        "reference_answer": "The figure shows a post-event slowdown in non-hydrocarbon GDP growth followed by a gradual recovery."
    },
    {
        "query": "What does the fiscal risk assessment figure indicate about Qatar’s fiscal risk?",
        "relevant_pages": [54, 59],
        "reference_answer": "The fiscal risk assessment indicates that Qatar faces low to moderate medium-term fiscal risk."
    },
    {
        "query": "Does the document provide information on cryptocurrency regulation in Qatar?",
        "relevant_pages": [],
        "reference_answer": "The document does not provide information on cryptocurrency regulation in Qatar."
    },
    {
        "query": "Explain in detail how Qatar’s fiscal buffers help manage economic shocks and ensure long-term sustainability.",
        "relevant_pages": [14, 15, 16, 18],
        "reference_answer": "Fiscal buffers support shock absorption and help maintain long-term fiscal sustainability."
    },
    {
        "query": "Describe how macroeconomic stability in Qatar is supported by fiscal and monetary coordination.",
        "relevant_pages": [12, 14, 15],
        "reference_answer": "Macroeconomic stability is supported by prudent fiscal management and a stable monetary framework."
    },
    {
        "query": "Based on the report, explain the interaction between non-hydrocarbon growth and diversification efforts.",
        "relevant_pages": [6, 7, 8, 32, 33],
        "reference_answer": "Non-hydrocarbon growth is supported by diversification efforts and structural reforms."
    },
    {
        "query": "Summarize how financial sector resilience is assessed, including capital, liquidity, and oversight aspects.",
        "relevant_pages": [12, 24, 25, 26],
        "reference_answer": "Financial sector resilience is supported by strong capitalization, ample liquidity, and effective oversight."
    },
    {
        "query": "How do external buffers contribute to Qatar’s resilience according to the external sector assessment?",
        "relevant_pages": [41, 42, 43],
        "reference_answer": "External buffers support resilience by strengthening the external position and reducing vulnerabilities."
    },
    {
        "query": "Explain how structural reforms are expected to support private sector-led growth over the medium term.",
        "relevant_pages": [32, 33, 34],
        "reference_answer": "Structural reforms aim to improve productivity and enable private sector-led growth."
    },
    {
        "query": "Describe the medium-term risks to the economic outlook and how policy buffers mitigate them.",
        "relevant_pages": [9, 10, 14, 15],
        "reference_answer": "Medium-term risks are mitigated by strong policy buffers and prudent macroeconomic management."
    },
    {
        "query": "How does the report link human capital development with long-term growth objectives?",
        "relevant_pages": [42, 43],
        "reference_answer": "Human capital development is linked to long-term growth by supporting productivity and diversification."
    },
    {
        "query": "What does the fiscal risk assessment imply when considered alongside fiscal policy discussions?",
        "relevant_pages": [14, 15, 54, 59],
        "reference_answer": "The assessment implies that fiscal risks are contained due to prudent fiscal policies."
    },
    {
        "query": "Explain how the charts and narrative together describe post-normalization economic trends.",
        "relevant_pages": [4, 5, 7, 49, 50],
        "reference_answer": "Charts and text together show normalization after earlier expansion with gradual recovery trends."
    },
    {
        "query": "Summarize how sustainability and climate considerations are integrated into development planning.",
        "relevant_pages": [43, 44],
        "reference_answer": "Sustainability and climate considerations are integrated as part of long-term development planning."
    },
    {
        "query": "How does the report assess medium-term growth prospects when combining outlook and reform sections?",
        "relevant_pages": [11, 12, 32, 33],
        "reference_answer": "Medium-term growth prospects are supported by reforms and stable macroeconomic conditions."
    },
    {
        "query": "Explain how the financial sector supports economic diversification according to the report.",
        "relevant_pages": [24, 25, 32, 33],
        "reference_answer": "The financial sector supports diversification by maintaining stability and enabling private activity."
    },
    {
        "query": "How do tables and figures together support the assessment of economic stability?",
        "relevant_pages": [35, 36, 37, 38, 49, 54],
        "reference_answer": "Tables and figures together indicate stable macroeconomic conditions and manageable risks."
    },
    {
        "query": "Based on the document, explain why Qatar is assessed as resilient to external shocks.",
        "relevant_pages": [12, 41, 42],
        "reference_answer": "Resilience is supported by strong buffers, prudent policies, and a solid external position."
    }
]