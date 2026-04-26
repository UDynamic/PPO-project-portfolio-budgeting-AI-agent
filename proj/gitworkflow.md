## commit messages
### syntax
for every commit message there is at least two identifiers:
1. modified messegaes: 
    `["created", "updated", "renamed", "moved", "deprecated", "removed"]`
2. file name.
3. specific acction.
    ```json
    {first import: "first import", 
    minor formatting or dictation: "polish", 
    "Archived", }
    ```
**syntax template:**
```bash
git add .
git commit -m "created README.md: imported first draft"
```

> for more than one modification in same commit:
> ```bash
> git add .
> git commit -m "created README.md: imported first draft 
> --- updated README.md: first polish "
> ```

---