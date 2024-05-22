import streamlit as st

# Define the formula and explanation
formula = r'\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla J(\theta)'
# explanation = (
#     r'Where: 
#     r'\newline'
#     r'\theta_{\text{new}}: Updated value of the parameter. '
#     r'\newline'
#     r'\theta_{\text{old}}: Previous value of the parameter. '
#     r'\newline'
#     r'\eta: Learning rate. '
#     r'\newline'
#     r'\nabla J(\theta): Gradient of the cost function with respect to $\theta$, also known as the slope or gradient. '
#     r'\newline'
#     r'In this context, $\nabla J(\theta)$ represents the slope or gradient of the cost function $J(\theta)$ with respect to the parameter $\theta$. '
#     r'\newline'
#     r'When we update the parameter $\theta$, we move it in the direction opposite to the gradient, scaled by the learning rate $\eta$.'
# )

# Display the formula and explanation
st.latex(formula)
st.latex(r'\theta_{\text{new}}: Updated\: value\:  of\:  the\: parameter.\quad  ')
st.latex(r'\theta_{\text{old}}: Previous\: value\: of\: the\: parameter.\: ')
st.latex(r'\eta: Learning\: rate. ')
st.latex(r'\nabla J(\theta) is\: just\: Gradient\: of\: function\: at\: that\: point\: also\: known\: as\: slope ')


st.latex(r'\theta_{\text{new}}\: - \: \theta_{\text{old}} \: = \: small')