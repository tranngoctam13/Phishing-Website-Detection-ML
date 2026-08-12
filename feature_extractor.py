import re
import socket
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import whois
from datetime import datetime
import pandas as pd


def extract_features(url):
    # Danh sách 30 cột đặc trưng giống hệt như file CSV của bạn
    feature_names = [
        'having_IPhaving_IP_Address', 'URLURL_Length', 'Shortining_Service',
        'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
        'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
        'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
        'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
        'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
        'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
        'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
    ]

    # Khởi tạo mặc định: Mọi yếu tố ban đầu được xem là An toàn (giá trị 1)
    features = {col: 1 for col in feature_names}

    # Tiền xử lý URL
    if not url.startswith('http'):
        url = 'http://' + url
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    # -------------------------------------------------------------
    # 1. CÁC ĐẶC TRƯNG TỪ CẤU TRÚC URL (Lexical Features)
    # -------------------------------------------------------------
    # Kiểm tra xem có dùng IP thay cho tên miền không
    try:
        socket.inet_aton(domain)
        features['having_IPhaving_IP_Address'] = -1
    except socket.error:
        pass

    # Chiều dài URL (Quá dài thường là lừa đảo để giấu domain thật)
    if len(url) >= 75:
        features['URLURL_Length'] = -1
    elif len(url) >= 54:
        features['URLURL_Length'] = 0

    # Dùng dịch vụ rút gọn link (bit.ly, tinyurl...)
    shorteners = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd"
    if re.search(shorteners, domain): features['Shortining_Service'] = -1

    # Chứa ký tự @ (Đánh lừa trình duyệt)
    if '@' in url: features['having_At_Symbol'] = -1

    # Chuyển hướng '//' sai vị trí
    if url.rfind('//') > 7: features['double_slash_redirecting'] = -1

    # Có dấu gạch ngang '-' trong tên miền (Ví dụ: techcom-bank.com)
    if '-' in domain: features['Prefix_Suffix'] = -1

    # Quá nhiều Sub-domain (Ví dụ: login.secure.facebook.com)
    subdomains = domain.split('.')
    if 'www' in subdomains: subdomains.remove('www')
    if len(subdomains) == 2:
        features['having_Sub_Domain'] = 0
    elif len(subdomains) > 2:
        features['having_Sub_Domain'] = -1

    # Giao thức HTTPS (Cơ bản)
    if not url.startswith('https'): features['SSLfinal_State'] = -1
    if 'https' in domain: features['HTTPS_token'] = -1

    # -------------------------------------------------------------
    # 2. CÁC ĐẶC TRƯNG MẠNG & WHOIS (Network Features)
    # -------------------------------------------------------------
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list): creation_date = creation_date[0]
        if creation_date:
            age_days = (datetime.now() - creation_date).days
            if age_days < 180: features['age_of_domain'] = -1  # Tên miền mới lập < 6 tháng
    except:
        features['DNSRecord'] = -1  # Không tra cứu được WHOIS -> Rất đáng ngờ

    # -------------------------------------------------------------
    # 3. CÁC ĐẶC TRƯNG NỘI DUNG HTML (HTML & JavaScript Features)
    # -------------------------------------------------------------
    try:
        # Tải mã nguồn trang web về (timeout 3 giây để web không bị treo)
        response = requests.get(url, timeout=3)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        # Chứa Iframe ẩn
        if len(soup.find_all('iframe', frameBorder="0")) > 0: features['Iframe'] = -1

        # Khóa chuột phải (Che giấu mã nguồn)
        if re.search(r"event\.button\s*==\s*2", html_content): features['RightClick'] = -1

        # Gửi form dữ liệu đến Email cá nhân (mailto:)
        if "mailto:" in html_content: features['Submitting_to_email'] = -1

        # Phân tích tỷ lệ liên kết đáng ngờ (Thẻ Anchor <a>)
        anchors = soup.find_all('a')
        bad_anchors = sum(1 for a in anchors if a.get('href', '') in ["", "#", "javascript:void(0)"])
        if len(anchors) > 0:
            ratio = bad_anchors / len(anchors)
            if ratio > 0.67:
                features['URL_of_Anchor'] = -1
            elif ratio > 0.31:
                features['URL_of_Anchor'] = 0
    except:
        # Nếu web chết hoặc chặn bot -> Ghi nhận bất thường
        features['web_traffic'] = -1

    # Biến đổi thành định dạng Pandas DataFrame để đưa vào AI
    return pd.DataFrame([features])