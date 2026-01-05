import streamlit as st
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datasets import get_data
from utils import mine_loss, true_mi_gaussian
from model_update import SimpleMineNetwork, ComplexMineNetwork 

st.set_page_config(page_title="MINE Project", layout="wide")
st.title("MINE: Mutual Information Neural Estimation")
st.markdown("---")
st.sidebar.header("Cấu hình")

data_type = st.sidebar.selectbox("Loại dữ liệu", ('gaussian', 'cubic', 'sine', 'circle'))

rho = 0.9
if data_type == 'gaussian':
    rho = st.sidebar.slider("Hệ số tương quan (Rho)", -0.99, 0.99, 0.9, 0.01)

dim = st.sidebar.selectbox("Số chiều (Dimension)", [1, 2, 5], index=0)
iter_count = st.sidebar.number_input("Số vòng lặp (Epoch)", value=1000)
lr = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f")
batch_size = st.sidebar.selectbox("Batch Size", [64, 128, 256, 512], index=2)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Dữ liệu đầu vào")
    x_vis, y_vis = get_data(300, 1, data_type, rho) 
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x_vis, y_vis, s=15, alpha=0.6, color='#2ecc71')
    ax.set_title(f"Distribution: {data_type.upper()}")
    st.pyplot(fig)

with col_right:
    st.subheader("2. Kết quả so sánh")
    if data_type == 'gaussian':
        theory_mi = true_mi_gaussian(rho, dim)
        st.metric("MI Lý thuyết (True MI)", f"{theory_mi:.4f}")
    else:
        st.info("Dữ liệu Phi tuyến: Không có True MI. Chúng ta so sánh xem mô hình nào hội tụ cao hơn.")
    
    start_btn = st.button("CHẠY DEMO", type="primary", use_container_width=True)

if start_btn:
    model_simple = SimpleMineNetwork(input_dim=dim*2, hidden_size=16)
    model_complex = ComplexMineNetwork(input_dim=dim*2, hidden_size=128)

    opt_simple = optim.Adam(model_simple.parameters(), lr=lr)
    opt_complex = optim.Adam(model_complex.parameters(), lr=lr)

    chart_placeholder = st.empty()
    progress_bar = st.progress(0)

    hist_simple = []
    hist_complex = []
    
    for i in range(iter_count):
        x, y = get_data(batch_size, dim, data_type, rho)
        x_t = torch.FloatTensor(x)
        y_t = torch.FloatTensor(y)

        opt_simple.zero_grad()
        loss_s, mi_s = mine_loss(model_simple, x_t, y_t)
        loss_s.backward()
        opt_simple.step()
        hist_simple.append(mi_s.item())

        opt_complex.zero_grad()
        loss_c, mi_c = mine_loss(model_complex, x_t, y_t)
        loss_c.backward()
        opt_complex.step()
        hist_complex.append(mi_c.item())

        if i % 20 == 0 or i == iter_count - 1:
            progress_bar.progress((i+1)/iter_count)
            
            fig2, ax2 = plt.subplots(figsize=(10, 4))

            ax2.plot(hist_simple, label='Simple Model (1 Layer)', color='orange', linewidth=1.5, alpha=0.8)

            ax2.plot(hist_complex, label='Complex Model (4 Layers)', color='red', linewidth=1.5)

            if data_type == 'gaussian':
                ax2.axhline(theory_mi, color='blue', linestyle='--', label='True MI')
                
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Mutual Information")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title("So sánh tốc độ hội tụ")
            
            chart_placeholder.pyplot(fig2)

    st.success("Hoàn thành so sánh!")
    c1, c2 = st.columns(2)
    c1.metric("Simple Model", f"{np.mean(hist_simple[-50:]):.4f}")
    c2.metric("Complex Model", f"{np.mean(hist_complex[-50:]):.4f}")
