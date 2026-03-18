"""
Conversational AI for Instant Business Intelligence Dashboards
A Streamlit app that generates interactive dashboards from natural language queries
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import os
from typing import Dict, Any, Optional, Tuple
import json

# Optional: Google Gemini API support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dashboard look
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


class QueryProcessor:
    """Process natural language queries and convert them to data operations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = df.columns.tolist()
        self.numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.text_columns = df.select_dtypes(include=['object']).columns.tolist()
        self.date_columns = self._detect_date_columns()
    
    def _detect_date_columns(self):
        """Detect columns that might contain date data"""
        date_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(self.df[col])
                    date_cols.append(col)
                except:
                    pass
        return date_cols
    
    def _find_column_match(self, search_terms: list) -> Optional[str]:
        """Find the best matching column for given search terms"""
        for term in search_terms:
            for col in self.columns:
                if term.lower() in col.lower():
                    return col
        return None
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the user query and determine the appropriate visualization"""
        query_lower = query.lower()
        result = {
            'chart_type': None,
            'data': None,
            'title': query,
            'error': None
        }
        
        try:
            # Detect chart type and analysis needed
            if any(word in query_lower for word in ['trend', 'over time', 'monthly', 'daily', 'yearly']):
                result['chart_type'] = 'line'
                result['data'] = self._process_time_series(query)
            
            elif any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs']):
                result['chart_type'] = 'bar'
                result['data'] = self._process_comparison(query)
            
            elif any(word in query_lower for word in ['distribution', 'breakdown', 'pie', 'proportion']):
                result['chart_type'] = 'pie'
                result['data'] = self._process_distribution(query)
            
            elif any(word in query_lower for word in ['top', 'bottom', 'best', 'worst']):
                result['chart_type'] = 'bar'
                result['data'] = self._process_ranking(query)
            
            elif any(word in query_lower for word in ['relationship', 'correlation', 'scatter']):
                result['chart_type'] = 'scatter'
                result['data'] = self._process_correlation(query)
            
            else:
                # Default to bar chart for general queries
                result['chart_type'] = 'bar'
                result['data'] = self._process_general(query)
            
            if result['data'] is None or result['data'].empty:
                result['error'] = "Could not generate visualization from the given query. Please try rephrasing or check if the required columns exist in your dataset."
        
        except Exception as e:
            result['error'] = f"Error processing query: {str(e)}"
        
        return result
    
    def _process_time_series(self, query: str) -> pd.DataFrame:
        """Process time series queries"""
        # Find date column
        date_col = None
        if self.date_columns:
            date_col = self.date_columns[0]
        else:
            for col in self.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
        
        if not date_col:
            return pd.DataFrame()
        
        # Find value column
        value_col = self._find_numeric_column_from_query(query)
        if not value_col and self.numeric_columns:
            value_col = self.numeric_columns[0]
        
        # Convert to datetime if needed
        df_copy = self.df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        
        # Group by month/day/year based on query
        if 'monthly' in query.lower():
            df_copy['period'] = df_copy[date_col].dt.strftime('%Y-%m')
        elif 'daily' in query.lower():
            df_copy['period'] = df_copy[date_col].dt.strftime('%Y-%m-%d')
        elif 'yearly' in query.lower():
            df_copy['period'] = df_copy[date_col].dt.strftime('%Y')
        else:
            df_copy['period'] = df_copy[date_col].dt.strftime('%Y-%m')
        
        # Group and aggregate
        result = df_copy.groupby('period')[value_col].sum().reset_index()
        result.columns = ['Date', value_col]
        return result
    
    def _process_comparison(self, query: str) -> pd.DataFrame:
        """Process comparison queries"""
        # Find category column
        category_col = self._find_text_column_from_query(query)
        if not category_col and self.text_columns:
            category_col = self.text_columns[0]
        
        # Find value column
        value_col = self._find_numeric_column_from_query(query)
        if not value_col and self.numeric_columns:
            value_col = self.numeric_columns[0]
        
        if not category_col or not value_col:
            return pd.DataFrame()
        
        # Group and aggregate
        result = self.df.groupby(category_col)[value_col].sum().reset_index()
        result = result.sort_values(value_col, ascending=False).head(10)
        return result
    
    def _process_distribution(self, query: str) -> pd.DataFrame:
        """Process distribution/pie chart queries"""
        # Find category column
        category_col = self._find_text_column_from_query(query)
        if not category_col and self.text_columns:
            category_col = self.text_columns[0]
        
        # Find value column
        value_col = self._find_numeric_column_from_query(query)
        if not value_col and self.numeric_columns:
            value_col = self.numeric_columns[0]
        
        if not category_col or not value_col:
            return pd.DataFrame()
        
        # Group and aggregate
        result = self.df.groupby(category_col)[value_col].sum().reset_index()
        result = result.sort_values(value_col, ascending=False).head(8)  # Limit to 8 slices for clarity
        return result
    
    def _process_ranking(self, query: str) -> pd.DataFrame:
        """Process ranking queries (top/bottom)"""
        # Extract number from query
        numbers = re.findall(r'\d+', query)
        n = int(numbers[0]) if numbers else 5
        
        # Find columns
        category_col = self._find_text_column_from_query(query)
        value_col = self._find_numeric_column_from_query(query)
        
        if not category_col and self.text_columns:
            category_col = self.text_columns[0]
        if not value_col and self.numeric_columns:
            value_col = self.numeric_columns[0]
        
        if not category_col or not value_col:
            return pd.DataFrame()
        
        # Group and sort
        result = self.df.groupby(category_col)[value_col].sum().reset_index()
        
        if 'bottom' in query.lower() or 'worst' in query.lower():
            result = result.nsmallest(n, value_col)
        else:
            result = result.nlargest(n, value_col)
        
        return result
    
    def _process_correlation(self, query: str) -> pd.DataFrame:
        """Process correlation/scatter plot queries"""
        if len(self.numeric_columns) < 2:
            return pd.DataFrame()
        
        # Take first two numeric columns for scatter plot
        x_col = self.numeric_columns[0]
        y_col = self.numeric_columns[1]
        
        result = self.df[[x_col, y_col]].dropna()
        return result
    
    def _process_general(self, query: str) -> pd.DataFrame:
        """Process general queries"""
        # Try to identify relevant columns from the query
        category_col = self._find_text_column_from_query(query)
        value_col = self._find_numeric_column_from_query(query)
        
        if not category_col and self.text_columns:
            category_col = self.text_columns[0]
        if not value_col and self.numeric_columns:
            value_col = self.numeric_columns[0]
        
        if not category_col or not value_col:
            return pd.DataFrame()
        
        # Default aggregation
        result = self.df.groupby(category_col)[value_col].sum().reset_index()
        result = result.sort_values(value_col, ascending=False).head(10)
        return result
    
    def _find_numeric_column_from_query(self, query: str) -> Optional[str]:
        """Find numeric column mentioned in query"""
        query_words = query.lower().split()
        for col in self.numeric_columns:
            col_words = col.lower().replace('_', ' ').split()
            if any(word in query_words for word in col_words):
                return col
        return None
    
    def _find_text_column_from_query(self, query: str) -> Optional[str]:
        """Find text column mentioned in query"""
        query_words = query.lower().split()
        for col in self.text_columns:
            col_words = col.lower().replace('_', ' ').split()
            if any(word in query_words for word in col_words):
                return col
        return None


class ChartGenerator:
    """Generate interactive Plotly charts based on processed data"""
    
    @staticmethod
    def create_chart(chart_type: str, data: pd.DataFrame, title: str) -> go.Figure:
        """Create appropriate chart based on type and data"""
        
        if data.empty:
            return None
        
        # Color scheme for consistency
        colors = px.colors.qualitative.Set3
        
        if chart_type == 'line':
            fig = px.line(
                data, 
                x=data.columns[0], 
                y=data.columns[1],
                title=title,
                markers=True,
                color_discrete_sequence=[colors[0]]
            )
            fig.update_traces(line=dict(width=3))
            
        elif chart_type == 'bar':
            fig = px.bar(
                data, 
                x=data.columns[0], 
                y=data.columns[1],
                title=title,
                color=data.columns[1],
                color_continuous_scale='viridis'
            )
            
        elif chart_type == 'pie':
            fig = px.pie(
                data, 
                values=data.columns[1], 
                names=data.columns[0],
                title=title,
                color_discrete_sequence=colors
            )
            
        elif chart_type == 'scatter':
            fig = px.scatter(
                data, 
                x=data.columns[0], 
                y=data.columns[1],
                title=title,
                trendline="ols",
                color_discrete_sequence=[colors[0]]
            )
            
        else:
            # Default to bar chart
            fig = px.bar(
                data, 
                x=data.columns[0], 
                y=data.columns[1],
                title=title,
                color_discrete_sequence=[colors[0]]
            )
        
        # Update layout for better appearance
        fig.update_layout(
            showlegend=True if chart_type == 'pie' else False,
            hovermode='x unified' if chart_type == 'line' else 'closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(l=40, r=40, t=60, b=40),
            height=450
        )
        
        # Update axes
        if chart_type != 'pie':
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig


def process_with_gemini(query: str, df_info: Dict) -> str:
    """Process query with Google Gemini API for better understanding"""
    if not GEMINI_AVAILABLE:
        return query
    
    api_key = st.session_state.get('gemini_api_key', '')
    if not api_key:
        return query
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Given a dataset with columns: {', '.join(df_info['columns'])}
        Numeric columns: {', '.join(df_info['numeric_columns'])}
        Text columns: {', '.join(df_info['text_columns'])}
        
        User query: "{query}"
        
        Rewrite this query to be more specific for data analysis.
        Keep it concise and mention specific column names if relevant.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return query


def main():
    """Main application logic"""
    
    # Header
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🤖 AI Business Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Transform your data into insights with natural language</p>", unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Optional Gemini API configuration
        if GEMINI_AVAILABLE:
            use_gemini = st.checkbox("Use Google Gemini AI", value=False)
            if use_gemini:
                api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key for enhanced query understanding")
                if api_key:
                    st.session_state['gemini_api_key'] = api_key
                    st.success("✅ Gemini API configured")
        else:
            st.info("📌 Install google-generativeai for enhanced AI capabilities")
        
        st.markdown("---")
        st.markdown("### 📊 Sample Queries")
        st.markdown("""
        - Show monthly sales revenue
        - Compare sales by category
        - Top 5 products by revenue
        - Distribution of customers by region
        - Show sales trend over time
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("Built with Streamlit, Pandas & Plotly")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze"
        )
        
        if uploaded_file is not None:
            # Load and display data info
            # Detect BMW CSV metadata prefix bug and skip rows
            if uploaded_file.name == 'BMW Vehicle Inventory.csv':
                df = pd.read_csv(uploaded_file, encoding='latin1', skiprows=10)
            else:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            st.session_state['df'] = df
            
            st.success(f"✅ Data loaded successfully!")
            st.markdown(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
            
            # Show column information
            with st.expander("📋 Dataset Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
            with st.expander("📊 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Unique': df.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
    
    with col2:
        if 'df' in st.session_state:
            st.markdown("### 💬 Ask Your Business Question")
            
            # Query input
            user_query = st.text_area(
                "Enter your question:",
                placeholder="e.g., 'Show monthly sales by region' or 'Compare revenue across product categories'",
                height=100
            )
            
            # Generate button
            if st.button("🚀 Generate Dashboard", type="primary", use_container_width=True):
                if user_query:
                    with st.spinner("🔍 Analyzing your query and generating insights..."):
                        # Process with Gemini if available
                        if GEMINI_AVAILABLE and st.session_state.get('gemini_api_key'):
                            df_info = {
                                'columns': st.session_state['df'].columns.tolist(),
                                'numeric_columns': st.session_state['df'].select_dtypes(include=['int64', 'float64']).columns.tolist(),
                                'text_columns': st.session_state['df'].select_dtypes(include=['object']).columns.tolist()
                            }
                            enhanced_query = process_with_gemini(user_query, df_info)
                            st.info(f"🤖 Enhanced query: {enhanced_query}")
                        else:
                            enhanced_query = user_query
                        
                        # Process query
                        processor = QueryProcessor(st.session_state['df'])
                        result = processor.analyze_query(enhanced_query)
                        
                        if result['error']:
                            st.error(f"❌ {result['error']}")
                        elif result['data'] is not None and not result['data'].empty:
                            # Generate chart
                            fig = ChartGenerator.create_chart(
                                result['chart_type'],
                                result['data'],
                                result['title']
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show data table
                                with st.expander("📊 View Data Table"):
                                    st.dataframe(result['data'], use_container_width=True)
                                
                                # Show metrics
                                if len(result['data']) > 0 and len(result['data'].columns) > 1:
                                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                                    
                                    value_col = result['data'].columns[1]
                                    with col_metric1:
                                        st.metric("Total", f"{result['data'][value_col].sum():,.0f}")
                                    with col_metric2:
                                        st.metric("Average", f"{result['data'][value_col].mean():,.0f}")
                                    with col_metric3:
                                        st.metric("Count", f"{len(result['data']):,}")
                        else:
                            st.warning("⚠️ No data could be generated for this query. Please try rephrasing or check your dataset columns.")
                else:
                    st.warning("Please enter a question to analyze your data.")
        else:
            st.info("👆 Please upload a CSV file to get started")


if __name__ == "__main__":
    main()
