import streamlit as st
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Import từ các module vệ tinh
from datasets import get_data
from utils import mine_loss, true_mi_gaussian
from models import MineNetwork

# --- CONFIG ---
st.set_page_config(page_title="MINE Project Demo", layout="wide")

st.title("MINE: Mutual Information Neural Estimation")
st.markdown("---")

# --- SIDEBAR CONTROL ---
st.sidebar.header("Cấu hình")

data_type = st.sidebar.selectbox("Loại dữ liệu", ('gaussian', 'cubic', 'sine', 'circle'))

rho = 0.9
if data_type == 'gaussian':
    rho = st.sidebar.slider("Hệ số tương quan (Rho)", -0.99, 0.99, 0.9, 0.01)

dim = st.sidebar.selectbox("Số chiều (Dimension)", [1, 2, 5, 10], index=0)
iter_count = st.sidebar.number_input("Số vòng lặp (Iterations)", value=1000)
lr = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f")

# --- MAIN DISPLAY ---
col_left, col_right = st.columns(2)

# 1. Visualize Data
with col_left:
    st.subheader("1. Dữ liệu đầu vào")
    x_vis, y_vis = get_data(300, 1, data_type, rho) # Luôn vẽ 1D để dễ nhìn
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x_vis, y_vis, s=15, alpha=0.6, color='#2ecc71')
    ax.set_title(f"Distribution: {data_type.upper()}")
    st.pyplot(fig)

# 2. Training Info
with col_right:
    st.subheader("2. Thông tin huấn luyện")
    if data_type == 'gaussian':
        theory_mi = true_mi_gaussian(rho, dim)
        st.metric("MI Lý thuyết (True MI)", f"{theory_mi:.4f}")
    else:
        st.info("Dữ liệu Phi tuyến: Không có công thức tính MI chính xác (Ground Truth). Chúng ta sẽ quan sát sự hội tụ.")
    
    start_btn = st.button("CHẠY DEMO", type="primary", use_container_width=True)

# --- TRAINING LOOP ---
if start_btn:
    model = MineNetwork(input_dim=dim*2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    progress_bar = st.progress(0)
    chart_placeholder = st.empty()
    mi_history = []
    
    for i in range(iter_count):
        # 1. Get data
        x, y = get_data(128, dim, data_type, rho)
        x_t = torch.FloatTensor(x)
        y_t = torch.FloatTensor(y)
        
        # 2. Train
        optimizer.zero_grad()
        loss, mi_est = mine_loss(model, x_t, y_t)
        loss.backward()
        optimizer.step()
        
        mi_history.append(mi_est.item())
        
        # 3. Update UI (mỗi 20 bước)
        if i % 20 == 0 or i == iter_count -1:
            progress_bar.progress((i+1)/iter_count)
            
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(mi_history, label='MINE Estimate', color='#e74c3c')
            if data_type == 'gaussian':
                ax2.axhline(theory_mi, color='blue', linestyle='--', label='True MI')
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Mutual Information")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            chart_placeholder.pyplot(fig2)

    st.success(f"Hoàn thành! Kết quả cuối cùng: {mi_history[-1]:.4f}")