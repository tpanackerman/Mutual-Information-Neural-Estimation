import streamlit as st
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from datasets import get_data
from utils import mine_loss, true_mi_gaussian
from models import MineNetwork

st.set_page_config(page_title="MINE Project Demo", layout="wide")
st.title("MINE: Mutual Information Neural Estimation")
st.markdown("---")

st.sidebar.header("Cấu hình")

data_type = st.sidebar.selectbox("Loại dữ liệu", ('gaussian', 'cubic', 'sine', 'circle'))

rho = 0.9
if data_type == 'gaussian':
    rho = st.sidebar.slider("Hệ số tương quan (Rho)", -0.99, 0.99, 0.9, 0.01)

dim = st.sidebar.selectbox("Số chiều (Dimension)", [1, 2, 5, 10], index=0)
iter_count = st.sidebar.number_input("Số vòng lặp (Epoch)", value=1000)
base_lr = st.sidebar.number_input("Base Learning Rate", value=0.001, format="%.4f")
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
    st.subheader("2. Thông tin huấn luyện")
    if data_type == 'gaussian':
        theory_mi = true_mi_gaussian(rho, dim)
        st.metric("MI Lý thuyết (True MI)", f"{theory_mi:.4f}")
    else:
        st.info("Dữ liệu Phi tuyến: Không có công thức tính MI chính xác. Chúng ta quan sát sự hội tụ.")
    
    st.write(f"Hệ thống sẽ chạy demo với 3 mức LR: {base_lr/10:.5f}, {base_lr:.4f}, và {base_lr*10:.4f}")
    start_btn = st.button("CHẠY DEMO", type="primary", use_container_width=True)

if start_btn:
    lrs = [base_lr / 10, base_lr, base_lr * 10]
    labels = [f'Low (LR={base_lr/10:.5f})', f'Base (LR={base_lr:.4f})', f'High (LR={base_lr*10:.4f})']
    colors = ['#27ae60', '#f39c12', '#c0392b'] 

    models = [MineNetwork(input_dim=dim*2) for _ in range(3)]
    opts = [optim.Adam(m.parameters(), lr=l) for m, l in zip(models, lrs)]
    
    histories = [[], [], []]

    progress_bar = st.progress(0)
    chart_placeholder = st.empty()
    
    for i in range(iter_count):
        x, y = get_data(batch_size, dim, data_type, rho)
        x_t = torch.FloatTensor(x)
        y_t = torch.FloatTensor(y)

        for idx in range(3):
            opts[idx].zero_grad()
            loss, mi_est = mine_loss(models[idx], x_t, y_t)
            loss.backward()
            opts[idx].step()
            
            histories[idx].append(mi_est.item())

        if i % 20 == 0 or i == iter_count - 1:
            progress_bar.progress((i+1)/iter_count)
            
            fig2, ax2 = plt.subplots(figsize=(10, 4))

            for idx in range(3):
                ax2.plot(histories[idx], label=labels[idx], color=colors[idx], alpha=0.8, linewidth=1.2)

            if data_type == 'gaussian':
                ax2.axhline(theory_mi, color='blue', linestyle='--', label='True MI')
            
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Mutual Information")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_title("So sánh tốc độ hội tụ với Learning Rate khác nhau")
            
            chart_placeholder.pyplot(fig2)
            plt.close(fig2) 

    st.success("Hoàn thành so sánh!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Low LR", f"{np.mean(histories[0][-50:]):.4f}")
    c2.metric("Base LR", f"{np.mean(histories[1][-50:]):.4f}")
    c3.metric("High LR", f"{np.mean(histories[2][-50:]):.4f}")
