from typing import TypedDict

class Person(TypedDict):
    name: str 
    age: int
    
new_person: Person = {'name': "Anekant", "Age": 10}
print(new_person)