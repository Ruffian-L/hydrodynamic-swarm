with open("build.rs", "r") as f:
    data = f.read()
with open("build.rs", "w") as f:
    f.write("fn main() {}\n")
