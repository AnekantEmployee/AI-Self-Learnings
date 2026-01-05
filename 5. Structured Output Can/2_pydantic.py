from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):
    name: str = "Anekant"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description='Indicates the cgpa')
    
# new_student = {}
# new_student = {'name': "Anekant", 'age':'3'} Supports implicit type conversion
new_student = {'name': "Anekant", 'age': 3, 'email': 'abc@gmail.com', 'cgpa': 8.32}
student = Student(**new_student)
print(type(student))
print(student)