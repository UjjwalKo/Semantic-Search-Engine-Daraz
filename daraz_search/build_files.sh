echo "BUILD START"
which python3
python3 --version
which pip3
pip3 --version

pip3 install -r requirements.txt
python3 manage.py collectstatic --no-input --clear
echo "BUILD END"
