import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import (
    clean_data,
    iqr_remove_outlier_rows,
    preprocess_dataset,
)

from sklearn.cluster import AgglomerativeClustering
from faiss_indexing import faiss_index
from clustering_model import full_clusterize, clusterize
from influence_signs import plot_cluster_profiles, importance_logreg


def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def main():
    st.set_page_config(
        page_title="Иерархическая кластеризация для промышленности",
        page_icon="🌵",
        layout="wide"
    )

    st.title("Иерархическая кластеризация для промышленности")

    st.markdown("""
    <style>
    div[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploadDropzone"]{
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 12px;
        background: rgba(127,127,127,0.06);
        padding: 18px 16px;
    }
    div[data-testid="stFileUploaderDropzone"] > div,
    div[data-testid="stFileUploadDropzone"] > div{
        display: flex; align-items: center; gap: 14px;
    }
    div[data-testid="stFileUploaderDropzone"]::before,
    div[data-testid="stFileUploadDropzone"]::before{
        content: "☁️"; font-size: 22px; opacity: .85; margin-right: 4px;
    }
    div[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploadDropzone"] button{
        border-radius: 10px; padding: 6px 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.session_state.setdefault('uploaded_name', None)
    st.session_state.setdefault('df_raw', None)
    st.session_state.setdefault('df_proc', None)
    st.session_state.setdefault('df_ready', None)
    st.session_state.setdefault('cluster_cols_w', [])
    st.session_state.setdefault('rows_to_show', None)

    st.markdown("**Загрузите файл**")
    uploaded = st.file_uploader(
        label="Выберите файл (CSV или Excel)",
        type=["csv", "xls", "xlsx", "xlsm"],
        label_visibility="collapsed",
    )
    st.caption("File type • CSV, Excel (.csv, .xls, .xlsx, .xlsm)")

    if uploaded is not None:
        name = uploaded.name.lower()
        df_loaded = None
        try:
            if name.endswith(".csv"):
                df_loaded = pd.read_csv(
                    uploaded,
                    sep=",",
                    decimal=",",
                    thousands="\u00A0",
                    engine="python"
                )
            elif name.endswith((".xlsx", ".xlsm")):
                df_loaded = pd.read_excel(uploaded, engine="openpyxl")
            elif name.endswith(".xls"):
                df_loaded = pd.read_excel(uploaded)
            else:
                st.error("Неподдерживаемый формат! Загрузите CSV или Excel.")
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")

        if df_loaded is not None:
            if st.session_state['uploaded_name'] != uploaded.name or st.session_state['df_raw'] is None:
                st.session_state['uploaded_name'] = uploaded.name
                st.session_state['df_raw'] = df_loaded
                st.session_state['df_proc'] = df_loaded.copy()
                st.session_state['df_ready'] = None
                st.session_state['cluster_cols_w'] = list(df_loaded.columns)
                st.session_state['rows_to_show'] = min(15, df_loaded.shape[0])

            df_raw = st.session_state['df_raw']
            df_proc = st.session_state['df_proc']

            st.success(f"Загружено: {uploaded.name} [{df_raw.shape[0]}×{df_raw.shape[1]}]")

            col_left, col_right = st.columns([7, 3])
            n_show_preview = min(df_raw.shape[0], df_raw.shape[1])
            with col_left:
                st.subheader("Предпросмотр данных")
                st.dataframe(df_raw.head(n_show_preview), use_container_width=True, hide_index=False)
            with col_right:
                st.subheader("Типы данных в столбцах")
                dtypes_df = pd.DataFrame({
                    "Столбец": df_raw.columns,
                    "Тип данных": df_raw.dtypes.astype(str).values
                }).reset_index(drop=True)
                st.dataframe(dtypes_df, use_container_width=True, hide_index=True, height=380)

            st.markdown("---")
            st.header("Предобработка данных")

            st.subheader("Переименование столбцов")
            c1, c2, c3 = st.columns([4, 6, 2])
            with c1:
                st.caption("Выберите столбец")
                col_to_rename = st.selectbox(
                    "Выберите столбец",
                    options=list(df_proc.columns),
                    label_visibility="collapsed",
                    key="rename_col"
                )
            with c2:
                st.caption("Новое имя")
                new_name = st.text_input(
                    "Новое имя",
                    placeholder="Введите новое имя столбца",
                    label_visibility="collapsed",
                    key="rename_new"
                )
            with c3:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                do_rename = st.button(
                    "Переименовать",
                    use_container_width=True,
                    type="secondary",
                    key="rename_btn"
                )
            if do_rename:
                nn = (new_name or "").strip()
                if nn == "":
                    st.warning("Введите новое имя столбца.")
                elif nn in df_proc.columns:
                    st.error(f"Столбец с именем «{nn}» уже существует.")
                else:
                    st.session_state['df_proc'] = df_proc.rename(columns={col_to_rename: nn})
                    st.session_state['df_ready'] = None
                    st.session_state['cluster_cols_w'] = list(st.session_state['df_proc'].columns)
                    _safe_rerun()

            st.subheader("Фильтр по дате")
            dc1, dc2, dc3 = st.columns([4, 6, 2])
            with dc1:
                st.caption("Выберите столбец с датой")
                date_col = st.selectbox(
                    "Столбец с датой",
                    options=list(st.session_state['df_proc'].columns),
                    index=0 if len(st.session_state['df_proc'].columns) else None,
                    label_visibility="collapsed",
                    key="date_col_select"
                )

            parsed = pd.to_datetime(
                st.session_state['df_proc'][date_col],
                errors="coerce",
                infer_datetime_format=True
            )
            try:
                parsed = parsed.dt.tz_localize(None)
            except Exception:
                pass

            valid = parsed.dropna()
            if valid.empty:
                st.info("Не получилось распознать даты в выбранном столбце.")
            else:
                min_date, max_date = valid.min().date(), valid.max().date()
                with dc2:
                    st.caption("Диапазон дат")
                    picked = st.date_input(
                        "Диапазон дат",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        label_visibility="collapsed",
                        key="date_range_input"
                    )
                with dc3:
                    st.markdown("&nbsp;", unsafe_allow_html=True)
                    do_filter = st.button(
                        "Отфильтровать",
                        use_container_width=True,
                        type="secondary",
                        key="date_filter_btn"
                    )
                if do_filter:
                    if isinstance(picked, tuple) and len(picked) == 2:
                        start_d, end_d = picked
                    else:
                        start_d = end_d = picked
                    mask = (parsed.dt.date >= start_d) & (parsed.dt.date <= end_d)
                    st.session_state['df_proc'] = st.session_state['df_proc'].loc[mask].copy()
                    st.session_state['df_ready'] = None
                    nmax = int(st.session_state['df_proc'].shape[0])
                    st.session_state['rows_to_show'] = min(
                        max(1, st.session_state.get('rows_to_show', 1)),
                        max(1, nmax)
                    )
                    _safe_rerun()

            st.subheader("Признаки для кластеризации")
            all_cols_now = list(st.session_state['df_proc'].columns)
            cur = [c for c in st.session_state.get('cluster_cols_w', []) if c in all_cols_now]
            if not cur:
                cur = all_cols_now[:]
            st.session_state['cluster_cols_w'] = cur

            st.multiselect("Выберите столбцы", options=all_cols_now, key="cluster_cols_w")
            sel_cols = st.session_state['cluster_cols_w']

            st.subheader("Препроцессинг данных (предобработка данных → IQR → encode/scale)")
            if st.button("Препроцессинг", type="primary", use_container_width=True, key="run_preprocessing"):
                try:
                    df_in = st.session_state['df_proc'][sel_cols] if sel_cols else st.session_state['df_proc']

                    df_clean = clean_data(df_in)
                    st.success("Предобработка данных пройдена")

                    df_iqr = iqr_remove_outlier_rows(df_clean)
                    st.success("IQR по числовым признакам реализован")

                    df_ready = preprocess_dataset(df_iqr)
                    st.success("Категориальные признаки закодированы, количественные признаки стандартизированы")

                    st.session_state['df_ready'] = df_ready

                    nmax = int(df_ready.shape[0])
                    st.session_state['rows_to_show'] = min(
                        max(1, st.session_state.get('rows_to_show', 15)),
                        max(1, nmax)
                    )
                except Exception as e:
                    st.error(f"Ошибка при предобработке: {e}")

            st.subheader("Превью датасета")
            df_preview = (
                st.session_state['df_ready']
                if st.session_state.get('df_ready') is not None
                else (st.session_state['df_proc'][sel_cols] if sel_cols else st.session_state['df_proc'])
            )

            nmax = int(df_preview.shape[0])
            default_n = min(15, nmax)

            if st.session_state['rows_to_show'] is None:
                st.session_state['rows_to_show'] = default_n
            else:
                st.session_state['rows_to_show'] = max(
                    1,
                    min(st.session_state['rows_to_show'], max(1, nmax))
                )

            rows_to_show = st.number_input(
                "Сколько строк показать",
                min_value=1,
                max_value=max(1, nmax),
                step=1,
                key="rows_to_show"
            )

            st.dataframe(
                df_preview.head(int(rows_to_show)),
                use_container_width=True,
                hide_index=False,
                height=420
            )

            total_rows, total_cols = df_preview.shape
            st.caption(f"Размер датасета: {total_rows} × {total_cols}")

            if st.session_state.get('df_ready') is not None and total_rows >= 2:
                st.markdown("---")
                st.header("Иерархическая кластеризация")

                df_ready_local = st.session_state['df_ready']
                n_rows = int(df_ready_local.shape[0])

                cc1, cc2 = st.columns([3, 3])
                with cc1:
                    n_clusters = st.number_input(
                        "Кол-во кластеров",
                        min_value=2,
                        max_value=max(2, min(50, n_rows)),
                        value=4,
                        step=1
                        #help="Рекомендуется 4 или 8, если хочется большей детализации. Число больше 9 может снижать интерпретируемость."
                    )
                with cc2:
                    max_levels = max(1, min(25, n_rows - 1))
                    dendro_levels = st.slider(
                        "Кол-во уровней в дендрограмме",
                        min_value=1,
                        max_value=max_levels,
                        value=min(5, max_levels),
                        step=1,
                        help="Сколько уровней показать в усечённой дендрограмме."
                    )

                run_btn = st.button("Кластеризировать", type="primary", use_container_width=True)

                if run_btn:
                    try:
                        X = df_ready_local.select_dtypes(include="number")

                        if X.shape[0] < int(n_clusters):
                            st.error("Число кластеров не может превышать число наблюдений.")
                        else:
                            tab_viz, tab_infl = st.tabs([
                                "Визуализация результатов кластеризации",
                                "Влияние признаков на кластеризацию"
                            ])

                            with tab_viz:
                                st.subheader("Дендрограмма")
                                with st.spinner("Строим дендрограмму…"):
                                    full_clusterize(X, p=int(dendro_levels))
                                    st.pyplot(plt.gcf(), use_container_width=True)

                                st.subheader("2D-визуализация иерархической кластеризации")
                                with st.spinner("Строим 2D-визуализации…"):
                                    clusterize(X, n_clusters=int(n_clusters))
                                    st.pyplot(plt.gcf(), use_container_width=True)

                            with tab_infl:
                                connectivity = faiss_index(X)

                                clusterer_finall = AgglomerativeClustering(
                                    n_clusters=int(n_clusters),
                                    linkage="ward",
                                    metric="euclidean",
                                    connectivity=connectivity,
                                    compute_distances=False
                                )
                                labels = clusterer_finall.fit_predict(X.values)

                                st.subheader("Тепловая карта средних значений признаков по кластерам")
                                with st.spinner("Считаем профили признаков…"):
                                    plot_cluster_profiles(X.copy(), labels)
                                    st.pyplot(plt.gcf(), use_container_width=True)

                                st.subheader("Важность признаков при кластеризации")
                                with st.spinner("Оцениваем важность признаков…"):
                                    importance_logreg(X.values, labels, feature_names=list(X.columns))
                                    st.pyplot(plt.gcf(), use_container_width=True)

                    except Exception as e:
                        st.error(f"Ошибка при кластеризации: {e}")


if __name__ == "__main__":
    main()
