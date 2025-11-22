import subprocess
import sys

def run_command(command, description):
    print(f"\n⏳ {description}...")
    try:
        # Run the command
        result = subprocess.run(
            command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Print output if any
        if result.stdout.strip():
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        # If git commit fails because there's nothing to commit, that's fine
        if "nothing to commit" in e.stdout or "nothing to commit" in e.stderr:
            print("ℹ️  Nothing to commit.")
            return True
            
        print(f"❌ Error during {description}:")
        print(e.stderr)
        return False

def git_push_automation():
    print("🚀 Starting Git Push Automation")

    # 1. Add all changes
    # This stages new files and modifications
    if not run_command("git add .", "Adding all changes"):
        return

    # 2. Check status to see if we need to commit
    status = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    
    if status.stdout.strip():
        # Ask for commit message
        commit_msg = input("\n💬 Enter commit message (Press Enter for 'Update code'): ").strip()
        if not commit_msg:
            commit_msg = "Update code"
            
        # 3. Commit
        if not run_command(f'git commit -m "{commit_msg}"', "Committing changes"):
            return
    else:
        print("\nℹ️  No new changes to commit.")

    # 4. Push
    if run_command("git push origin main", "Pushing to GitHub"):
        print("\n✅ Done! Code successfully pushed to GitHub.")

if __name__ == "__main__":
    git_push_automation()
