import numpy as np

def get_data(batch_size=64, dim=1, data_type='gaussian', rho=0.9):
    """
    Sinh cặp dữ liệu (x, y) dựa trên loại phân phối yêu cầu.
    """
    if data_type == 'gaussian':
        mean = np.zeros(2 * dim)
        cov = np.eye(2 * dim)
        cov[0:dim, dim:2*dim] = rho * np.eye(dim)
        cov[dim:2*dim, 0:dim] = rho * np.eye(dim)
        data = np.random.multivariate_normal(mean, cov, batch_size)
        return data[:, :dim], data[:, dim:]

    elif data_type == 'cubic':
        x = np.random.normal(0, 1, (batch_size, dim))
        noise = np.random.normal(0, 0.1, (batch_size, dim))
        y = x**3 + noise
        return x, y

    elif data_type == 'sine':
        x = np.random.normal(0, 2, (batch_size, dim))
        noise = np.random.normal(0, 0.1, (batch_size, dim))
        y = np.sin(x) + noise
        return x, y
    
    elif data_type == 'circle':
        angle = np.random.uniform(0, 2*np.pi, (batch_size, dim))
        r = 1 + np.random.normal(0, 0.05, (batch_size, dim))
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return x, y

    else:
        raise ValueError(f"Unknown data type: {data_type}")