import streamlit as st

# Progress tracker component
def progress_tracker(current_step):
    """
    Display a progress tracker for the app workflow
    
    Parameters:
    current_step (int): Current step index (0-based)
    """
    steps = ["User Profile", "Spending Input", "Recommendations", "Chat"]
    step_icons = ["👤", "💰", "💳", "🤖"]
    
    # Add custom CSS for the progress tracker
    st.markdown("""
    <style>
    .step-container {
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 8px;
        border-radius: 8px;
        margin: 5px;
        font-weight: bold;
    }
    .completed-step {
        background-color: #e6f7f2;
        color: #07b16b;
        border: 1px solid #07b16b;
    }
    .current-step {
        background-color: #f0f7fe;
        color: #1c83e1;
        border: 2px solid #1c83e1;
    }
    .upcoming-step {
        background-color: #f7f7f7;
        color: #9e9e9e;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create columns for the steps
    cols = st.columns(len(steps))
    
    # Display each step with appropriate styling
    for i, (step, icon) in enumerate(zip(steps, step_icons)):
        with cols[i]:
            if i < current_step:
                # Completed step
                st.markdown(f"""
                <div class="step-container completed-step">
                ✓ {icon} {step}
                </div>
                """, unsafe_allow_html=True)
            elif i == current_step:
                # Current step
                st.markdown(f"""
                <div class="step-container current-step">
                → {icon} {step}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Upcoming step
                st.markdown(f"""
                <div class="step-container upcoming-step">
                {icon} {step}
                </div>
                """, unsafe_allow_html=True)
    
    # Add a separator after the progress tracker
    st.divider()

# Navigation buttons with consistent styling
def nav_buttons(prev_page=None, next_page=None, next_condition=True, next_warning=None, home=False):
    """
    Create standardized navigation buttons
    
    Parameters:
    prev_page (str): Path to previous page
    next_page (str): Path to next page
    next_condition (bool): Condition that must be met to enable next button
    next_warning (str): Warning message if next_condition is False
    home (bool): Whether to include a home button
    """
    # Create a container with border and padding for navigation
    with st.container():
        st.markdown("""
        <style>
        .nav-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        </style>
        <div class="nav-container"></div>
        """, unsafe_allow_html=True)
        
        # Use more balanced columns depending on which buttons are shown
        if home:
            cols = st.columns([1.2, 0.8, 1.2])
        else:
            cols = st.columns([1, 2, 1])
        
        # Previous page button
        if prev_page:
            with cols[0]:
                if "User_Profile" in prev_page:
                    st.page_link(prev_page, label="← User Profile", icon="👤", use_container_width=True)
                elif "Spending_Input" in prev_page:
                    st.page_link(prev_page, label="← Spending Input", icon="💰", use_container_width=True)
                elif "Recommendations" in prev_page:
                    st.page_link(prev_page, label="← Recommendations", icon="💳", use_container_width=True)
                elif "Welcome.py" in prev_page:
                    st.page_link(prev_page, label="← Home", icon="🏠", use_container_width=True)
                else:
                    st.page_link(prev_page, label="← Back", icon="⬅️", use_container_width=True)
        
        # Home button (in middle)
        if home:
            with cols[1]:
                st.page_link("Welcome.py", label="Home", icon="🏠", use_container_width=True)
        
        # Next page button
        if next_page:
            with cols[2]:
                if next_condition:
                    if "User_Profile" in next_page:
                        st.page_link(next_page, label="User Profile →", icon="👤", use_container_width=True)
                    elif "Spending_Input" in next_page:
                        st.page_link(next_page, label="Spending Input →", icon="💰", use_container_width=True)
                    elif "Recommendations" in next_page:
                        st.page_link(next_page, label="Recommendations →", icon="💳", use_container_width=True)
                    elif "Chat" in next_page:
                        st.page_link(next_page, label="Chat with AI →", icon="🤖", use_container_width=True)
                    else:
                        st.page_link(next_page, label="Next →", icon="➡️", use_container_width=True)
                else:
                    # Add a more visual indication that this button is disabled
                    st.markdown(f"""
                    <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;text-align:center;color:#666;">
                    ⚠️ {next_warning or "Please complete this section before proceeding"}
                    </div>
                    """, unsafe_allow_html=True)

# Standard page header with title, icon, and description
def page_header(title, icon, description=None):
    """
    Create a standardized page header with title and optional description
    
    Parameters:
    title (str): Page title
    icon (str): Emoji icon for the page
    description (str, optional): Page description
    """
    # Add custom CSS for the header
    st.markdown("""
    <style>
    .page-header {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        margin-bottom: 20px;
    }
    .page-title {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    .page-description {
        opacity: 0.8;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create header with title and description
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">{icon} {title}</div>
        {f'<div class="page-description">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

# Helper for section headers
def section_header(title, description=None):
    """
    Create a standardized section header with title and optional description
    
    Parameters:
    title (str): Section title
    description (str, optional): Section description
    """
    # Add custom CSS for the section header
    st.markdown("""
    <style>
    .section-header {
        border-left: 4px solid #1c83e1;
        padding-left: 10px;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .section-description {
        opacity: 0.8;
        font-size: 1rem;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create section header with title and description
    st.markdown(f"""
    <div class="section-header">
        <div class="section-title">{title}</div>
        {f'<div class="section-description">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# Helper for subsection headers
def subsection_header(title, description=None):
    """
    Create a standardized subsection header with title and optional description
    
    Parameters:
    title (str): Subsection title
    description (str, optional): Subsection description
    """
    # Add custom CSS for the subsection header
    st.markdown("""
    <style>
    .subsection-header {
        border-left: 3px solid #07b16b;
        padding-left: 8px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .subsection-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subsection-description {
        opacity: 0.8;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create subsection header with title and description
    st.markdown(f"""
    <div class="subsection-header">
        <div class="subsection-title">{title}</div>
        {f'<div class="subsection-description">{description}</div>' if description else ''}
    </div>
    """, unsafe_allow_html=True) 