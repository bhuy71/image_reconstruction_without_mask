from collections import defaultdict, deque
from datetime import datetime

class Person:
    def __init__(self, code, dob, father_code, mother_code, is_alive, region_code):
        self.code = code
        self.dob = dob
        self.father_code = father_code
        self.mother_code = mother_code
        self.is_alive = is_alive
        self.region_code = region_code

class Database:
    def __init__(self):
        self.people = []
        self.people_by_code = {}
        self.birth_date_count = defaultdict(int)
        self.parent_child_map = defaultdict(list)

    def add_person(self, code, dob, father_code, mother_code, is_alive, region_code):
        person = Person(code, dob, father_code, mother_code, is_alive, region_code)
        self.people.append(person)
        self.people_by_code[code] = person
        self.birth_date_count[dob] += 1
        
        if father_code != "0000000":
            self.parent_child_map[father_code].append(code)
        if mother_code != "0000000":
            self.parent_child_map[mother_code].append(code)

    def number_people(self):
        return len(self.people)

    def number_people_born_at(self, date):
        return self.birth_date_count.get(date, 0)

    def most_alive_ancestor(self, code):
        if code not in self.people_by_code:
            return 0
        
        queue = deque([(code, 0)])
        most_distant_alive = None
        visited = set()
        
        while queue:
            current_code, generation = queue.popleft()
            if current_code in visited:
                continue
            visited.add(current_code)
            
            person = self.people_by_code.get(current_code)
            if not person:
                continue
            
            if person.is_alive == 'Y':
                most_distant_alive = generation
            
            father = person.father_code
            mother = person.mother_code
            
            if father != "0000000":
                queue.append((father, generation + 1))
            if mother != "0000000":
                queue.append((mother, generation + 1))
        
        return most_distant_alive if most_distant_alive is not None else 0

    def number_people_born_between(self, from_date, to_date):
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")
        count = 0
        
        for dob, cnt in self.birth_date_count.items():
            dob_dt = datetime.strptime(dob, "%Y-%m-%d")
            if from_dt <= dob_dt <= to_dt:
                count += cnt
                
        return count

    def max_unrelated_people(self):
        unrelated_set = set()
        
        for person in self.people:
            code = person.code
            if code in unrelated_set:
                continue
            
            unrelated_set.add(code)
            for child in self.parent_child_map[code]:
                unrelated_set.discard(child)
        
        return len(unrelated_set)

def main():
    database = Database()
    processing_queries = False
    results = []

    while True:
        line = input().strip()
        
        if line == "*":
            processing_queries = True
            continue
        elif line == "***":
            break

        if not processing_queries:
            code, dob, father_code, mother_code, is_alive, region_code = line.split()
            database.add_person(code, dob, father_code, mother_code, is_alive, region_code)
        else:
            parts = line.split()
            command = parts[0]
            
            if command == "NUMBER_PEOPLE":
                results.append(database.number_people())
            elif command == "NUMBER_PEOPLE_BORN_AT":
                date = parts[1]
                results.append(database.number_people_born_at(date))
            elif command == "MOST_ALIVE_ANCESTOR":
                code = parts[1]
                results.append(database.most_alive_ancestor(code))
            elif command == "NUMBER_PEOPLE_BORN_BETWEEN":
                from_date = parts[1]
                to_date = parts[2]
                results.append(database.number_people_born_between(from_date, to_date))
            elif command == "MAX_UNRELATED_PEOPLE":
                results.append(database.max_unrelated_people())

    print("\n".join(map(str, results)))

main()
