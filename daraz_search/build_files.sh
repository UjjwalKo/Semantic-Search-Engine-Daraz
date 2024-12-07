echo "BUILD START"
pip3.12.1 -m install -r requirements.txt
python3.12.1 manage.py collectstatic --no-input --clear
echo "BUILD END"