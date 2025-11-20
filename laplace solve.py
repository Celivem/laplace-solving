import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# 1. 設定頁面配置：更改標題為 "laplace可視化"
st.set_page_config(page_title="laplace可視化", layout="wide")

# 2. 更改主標題
st.title("laplace可視化")
st.markdown("支援 **自然數學輸入** (如 `x^2`, `2x`, `e^-x`) 與 **四邊無窮邊界模擬**。")

# 定義共用的繪圖函數
def plot_heatmap(data, title, xlabel="x", ylabel="y"):
    fig, ax = plt.subplots(figsize=(8, 6))
    # 顏色: 紅(高) -> 彩虹 -> 紫(低)
    im = ax.imshow(data, cmap='rainbow', origin='lower', 
                   extent=[0, 1, 0, 1], aspect='auto', interpolation='bilinear')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Potential (V)')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    return fig

# 智能解析函數
def smart_parse(input_str):
    if not input_str or input_str.strip() == "0":
        return None
    
    # 定義轉換規則：標準 + 隱式乘法 (2x -> 2*x) + 次方轉換 (^ -> **)
    transformations = (standard_transformations + 
                       (implicit_multiplication_application,) + 
                       (convert_xor,))
    
    # 定義本地變數對應，讓 'e' 對應到數學常數 E
    local_dict = {'e': sp.E, 'pi': sp.pi}
    
    try:
        # 解析
        expr = parse_expr(input_str, transformations=transformations, local_dict=local_dict)
        return expr
    except Exception as e:
        st.error(f"無法解析輸入 '{input_str}'，請檢查語法。錯誤: {e}")
        return None

# --- 側邊欄 ---
mode = st.sidebar.radio("選擇求解模式", ["數值解 (模擬任意邊界)", "解析解 (公式推導+疊加)"])

# ==========================================
# 模式一：數值解 (FDM)
# ==========================================
if mode == "數值解 (模擬任意邊界)":
    st.header("數值解 (Finite Difference Method)")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("邊界條件設定")
        def input_boundary(label, default_val):
            is_inf = st.checkbox(f"{label} 設為無窮遠", key=f"inf_{label}")
            if not is_inf:
                val = st.number_input(f"{label} 電位 (V)", value=default_val, key=f"v_{label}")
            else:
                val = 0.0
                st.caption(f"*{label} 為開放邊界*")
            return is_inf, val

        top_inf, top_v = input_boundary("上邊界 (Top)", 10.0)
        bottom_inf, bottom_v = input_boundary("下邊界 (Bottom)", 0.0)
        left_inf, left_v = input_boundary("左邊界 (Left)", 0.0)
        right_inf, right_v = input_boundary("右邊界 (Right)", 0.0)
        
        st.markdown("---")
        grid_size = st.slider("解析度", 30, 100, 50)
        iterations = st.slider("迭代次數", 1000, 10000, 3000)

    with col2:
        if st.button("開始計算"):
            with st.spinner("計算中..."):
                core_size = grid_size 
                pad = core_size * 3
                pad_top = pad if top_inf else 0
                pad_bottom = pad if bottom_inf else 0
                pad_left = pad if left_inf else 0
                pad_right = pad if right_inf else 0
                
                total_h = pad_top + core_size + pad_bottom
                total_w = pad_left + core_size + pad_right
                V = np.zeros((total_h, total_w))
                
                row_start, row_end = pad_bottom, pad_bottom + core_size
                col_start, col_end = pad_left, pad_left + core_size
                
                for k in range(iterations):
                    V_old = V.copy()
                    V[1:-1, 1:-1] = 0.25 * (V_old[0:-2, 1:-1] + V_old[2:, 1:-1] + 
                                            V_old[1:-1, 0:-2] + V_old[1:-1, 2:])
                    
                    if not top_inf: V[row_end-1, col_start:col_end] = top_v
                    else: V[-1, :] = 0.0 
                    if not bottom_inf: V[row_start, col_start:col_end] = bottom_v
                    else: V[0, :] = 0.0
                    if not left_inf: V[row_start:row_end, col_start] = left_v
                    else: V[:, 0] = 0.0
                    if not right_inf: V[row_start:row_end, col_end-1] = right_v
                    else: V[:, -1] = 0.0

                V_view = V[row_start:row_end, col_start:col_end]
                st.pyplot(plot_heatmap(V_view, "Potential Distribution"))
                st.success("完成")

# ==========================================
# 模式二：解析解 (SymPy) - 智能輸入版
# ==========================================
elif mode == "解析解 (公式推導+疊加)":
    st.header("解析解 (Analytical Solution)")
    st.markdown("輸入數學式 (如 `x^2`, `sin(pi x)`, `e^-x`)，系統將自動推導並疊加四邊貢獻。")

    col1, col2 = st.columns(2)
    with col1:
        str_top = st.text_input("上邊界 V(x, 1) =", value="10")
        str_bottom = st.text_input("下邊界 V(x, 0) =", value="0")
    with col2:
        str_left = st.text_input("左邊界 V(0, y) =", value="0")
        str_right = st.text_input("右邊界 V(1, y) =", value="0")

    if st.button("推導公式並生成圖片"):
        try:
            with st.spinner("正在解析數學式並進行符號運算..."):
                x, y, n = sp.symbols('x y n')
                pi = sp.pi
                
                def solve_boundary_smart(input_str, b_type):
                    f_expr = smart_parse(input_str)
                    if f_expr is None: return None
                    
                    denom = sp.sinh(n * pi)
                    if b_type == 'top':
                        An = 2 * sp.integrate(f_expr * sp.sin(n * pi * x), (x, 0, 1))
                        return (An * sp.sin(n * pi * x) * sp.sinh(n * pi * y)) / denom
                    elif b_type == 'bottom':
                        An = 2 * sp.integrate(f_expr * sp.sin(n * pi * x), (x, 0, 1))
                        return (An * sp.sin(n * pi * x) * sp.sinh(n * pi * (1 - y))) / denom
                    elif b_type == 'left':
                        # 左邊界是 f(y)，變數替換 x->y
                        f_y = f_expr.subs(x, y) 
                        An = 2 * sp.integrate(f_y * sp.sin(n * pi * y), (y, 0, 1))
                        return (An * sp.sin(n * pi * y) * sp.sinh(n * pi * (1 - x))) / denom
                    elif b_type == 'right':
                        f_y = f_expr.subs(x, y)
                        An = 2 * sp.integrate(f_y * sp.sin(n * pi * y), (y, 0, 1))
                        return (An * sp.sin(n * pi * y) * sp.sinh(n * pi * x)) / denom

                # 計算各部分
                terms = []
                for s, t in [(str_top, 'top'), (str_bottom, 'bottom'), 
                             (str_left, 'left'), (str_right, 'right')]:
                    res = solve_boundary_smart(s, t)
                    if res is not None: terms.append(res)

                if not terms:
                    st.warning("所有邊界皆為 0")
                else:
                    # 總公式
                    V_total = sum(terms)
                    latex_code = sp.latex(V_total)
                    st.success("推導完成！")
                    
                    # 顯示公式 (如果太長就截斷顯示)
                    if len(latex_code) > 1000:
                        st.latex("V(x,y) = \\sum ... (公式過長省略顯示)")
                    else:
                        st.latex(f"V(x,y) = \\sum_{{n=1}}^{{\\infty}} \\left( {latex_code} \\right)")

                    # --- 繪圖區 ---
                    st.markdown("---")
                    st.subheader("解析解視覺化")
                    
                    n_terms = 20
                    grid_res = 100
                    xv = np.linspace(0, 1, grid_res)
                    yv = np.linspace(0, 1, grid_res)
                    X, Y = np.meshgrid(xv, yv)
                    V_sum = np.zeros_like(X)
                    
                    # 編譯成 NumPy 函數
                    x_sym, y_sym, n_sym = sp.symbols('x y n')
                    func_numpy = sp.lambdify((n_sym, x_sym, y_sym), V_total, 'numpy')
                    
                    progress_bar = st.progress(0)
                    for i in range(1, n_terms + 1):
                        try:
                            Z = func_numpy(i, X, Y)
                            V_sum += np.nan_to_num(np.array(Z, dtype=float))
                        except: pass
                        progress_bar.progress(i / n_terms)
                    
                    st.pyplot(plot_heatmap(V_sum, f"Analytical Result (Top 20 terms)"))

        except Exception as e:
            st.error(f"發生錯誤: {e}")
