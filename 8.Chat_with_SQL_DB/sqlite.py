import sqlite3


##Connect to sqlite

connection = sqlite3.connect("student.db")

##Create a cursor object to insert record , create table
cursor= connection.cursor()

##Create a table
table_info= """
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25), SECTION VARCHAR(25), MARK INT)
"""

cursor.execute(table_info)

#Insert some more records
cursor.execute('''Insert Into STUDENT values('Krish','Data science','A',90)''')
cursor.execute('''Insert Into STUDENT values('John','Data science','B',100)''')
cursor.execute('''Insert Into STUDENT values('Mukesh','Data science','A',86)''')
cursor.execute('''Insert Into STUDENT values('Jacob','Devops','A',50)''')
cursor.execute('''Insert Into STUDENT values('Siva',' Devops','A',35)''')


##Display all the records
print("The inserted records are")
data = cursor.execute('''Select * from STUDENT ''')

for row in data:
    print(row)
    
    
## commit your changes in the database
connection.commit()
connection.close()