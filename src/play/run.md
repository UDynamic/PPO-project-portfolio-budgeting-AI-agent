del src\db\database.db
del src\db\exports
python src\db\db_init.py
python src\env\configs\single_project.py
python src\play\play.py

python src\db\export.py 