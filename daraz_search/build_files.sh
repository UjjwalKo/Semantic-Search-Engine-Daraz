echo "BUILD START"
pip3 install -r requirements.txt
python3 manage.py collectstatic --no-input --clear
echo "BUILD END"
