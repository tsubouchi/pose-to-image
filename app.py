import streamlit as st
import mediapipe as mp
from PIL import Image
import io
from pose_extractor import extract_pose
from image_generator import generate_image

st.set_page_config(
    page_title="Pose to Image Generator",
    layout="wide"
)

st.title("ポーズから画像生成")

# Initialize session state for images and system prompt
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'pose_image' not in st.session_state:
    st.session_state.pose_image = None
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = """
    重要: この棒人間は正確な人体の座標系を表現しています。以下の指示を厳密に守って画像を生成してください。

    0. 人数の制御と構図:
    - 入力画像の棒人間は1人です
    - 出力画像でも必ず1人のみを生成すること
    - 背景の人物や反射した人物も含めて、画像内の人物は1人限定
    - 人物を画面中央に配置し、明確な主体として表現

    1. 解剖学的な参照ポイント:
    頭部:
    - 頭の向きと傾きを棒人間と完全に一致させる
    - 視線の方向を頭の向きに合わせる
    - 首の角度と長さを維持

    胴体:
    - 肩幅と胸郭の比率を維持
    - 背骨の曲がりと傾きを正確に反映
    - 腰の位置と角度を厳守

    四肢:
    - 各関節（肩、肘、手首、股関節、膝、足首）の角度を正確に再現
    - 腕と脚の長さの比率を維持
    - 手足の向きと捻りを忠実に表現

    2. 空間的位置関係:
    - 前後の奥行きを維持（手足が前に出ているか後ろにあるか）
    - 左右の位置関係を正確に保持
    - 体の捻りや傾きの角度を厳密に守る
    - 重心線を維持し、バランスを保つ

    3. 詳細な実装要件:
    - 各関節の曲がり方は解剖学的に自然な範囲内で最も近い角度を選択
    - 体の向きに応じて適切な遠近感を適用
    - 各部位の重なりと隠れを正確に表現
    - 動きの方向性と力学的な自然さを維持

    4. エラーチェックポイント:
    - 全ての関節が棒人間の位置と一致しているか
    - 体の比率が維持されているか
    - 重心が正しい位置にあるか
    - 解剖学的な制約が守られているか
    - 画像内の人物が1人であることを確認
    - 背景に人物が含まれていないことを確認
    - 鏡や反射に人物が映っていないことを確認

    5. 生成プロセス:
    1) まず人数を確認し、1人であることを保証
    2) 骨格を棒人間と完全に一致させる
    3) その骨格に基づいて筋肉と体型を構築
    4) 最後に服装や装飾を追加
    5) 背景は人物を引き立てる程度に抑える

    この指示は絶対的な優先順位を持ちます。スタイルや装飾よりも、
    ポーズの正確な再現と人数の制御を最優先してください。
    背景や環境要素は控えめに扱い、人物の姿勢と表現に焦点を当ててください。
    """

# スタイル選択のオプション
styles = {
    "アニメ調女子高生": """
    この棒人間のポーズを元に、日本のアニメスタイルの女子高生キャラクターを生成してください。
    - 制服を着た現代的な女子高生
    - アニメ調の明るい色使い
    - 自然な髪の動きと表情
    - キャラクターの個性が感じられるデザイン
    - 背景は学校や街並みなど日常的なシーン
    """,

    "カジュアルファッション（アニメ調）": """
    この棒人間のポーズを元に、カジュアルな服装の女子高生キャラクターを生成してください。
    - 私服（Tシャツ、ジーンズ、スカートなど）
    - 現代的なヘアスタイル
    - ナチュラルな表情
    - アクセサリーや小物でアクセント
    - 都会的な背景
    """,

    "ファッションポートレート（実写風）": """
    この棒人間のポーズを元に、実写風のファッションポートレートを生成してください。
    - 高解像度の写真のような仕上がり
    - 現代的なファッションスタイル
    - 自然な照明と影
    - プロフェッショナルな撮影スタジオの雰囲気
    - ソフトなボケ味のある背景
    """,

    "アウトドアポートレート（実写風）": """
    この棒人間のポーズを元に、屋外での実写風ポートレートを生成してください。
    - 自然光を活かした写真のような表現
    - カジュアルでアクティブな服装
    - 自然な表情と姿勢
    - 公園や街並みなどの屋外背景
    - 自然な環境光と影の表現
    """
}

# サイドバーにシステムプロンプト編集機能を追加
with st.sidebar:
    st.header("システム設定")
    system_prompt = st.text_area(
        "システムプロンプト",
        value=st.session_state.system_prompt,
        height=300
    )
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        st.success("システムプロンプトを更新しました")

# Top section: Drag & Drop
uploaded_file = st.file_uploader("画像をドラッグ＆ドロップしてください", type=['png', 'jpg', 'jpeg'])

# Style selection
selected_style = st.selectbox(
    "生成スタイルを選択",
    list(styles.keys())
)

# Create three columns for the images
col1, col2, col3 = st.columns(3)

if uploaded_file is not None:
    try:
        # Display original image
        with col1:
            st.header("元画像")
            image = Image.open(uploaded_file)
            st.session_state.original_image = image
            st.image(image, caption="Original Image")

        # Extract and display pose
        with col2:
            st.header("ポーズ抽出")
            pose_image = extract_pose(image)
            st.session_state.pose_image = pose_image
            st.image(pose_image, caption="Extracted Pose")

        # Generate and display new image
        with col3:
            st.header("生成画像")
            generated_image = generate_image(
                pose_image, 
                styles[selected_style],
                st.session_state.system_prompt
            )
            st.session_state.generated_image = generated_image
            if generated_image is not None:
                st.image(generated_image, caption="Generated Image")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

st.markdown("""
---
### 使い方:
1. 人物が写っている画像をドラッグ＆ドロップ
2. 生成したいスタイルを選択
   - アニメ調: アニメやイラスト風の画像を生成
   - 実写風: 写真のような現実的な画像を生成
3. 必要に応じてサイドバーでシステムプロンプトを調整
4. システムが自動的にポーズを抽出し、選択したスタイルで新しい画像を生成
""")