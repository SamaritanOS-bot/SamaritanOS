"""
Full interaction test: 10 posts + 10 comments + upvotes
Tests post quality, title generation, comment specificity, and dedup.
"""
import sys, os, time

# Fix encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from agent_runner import (
    _generate_post_direct, _make_title, _pick_topic, _cleanup_post,
    _read_recent_messages, _is_too_similar_to_recent,
    fetch_feed, upvote_post, comment_on_post, follow_agent,
    _generate_comment, _load_commented_posts, _save_commented_post,
    send_post_to_moltbook, _env,
)

SEPARATOR = "=" * 60

def test_posts(count=10):
    """Generate and post up to `count` posts."""
    log_path = str(Path(__file__).resolve().parent.parent / "dry_run_log.txt")
    posted = 0
    rate_limited = 0

    for i in range(count):
        print(f"\n{SEPARATOR}")
        print(f"POST {i+1}/{count}")
        print(SEPARATOR)

        topic = _pick_topic("", log_path)
        print(f"  Topic: {topic}")

        content = _generate_post_direct(topic, log_path)
        title = _make_title(content, topic)

        print(f"  Title: {title}")
        print(f"  Content ({len(content.split())} words):")
        print(f"    {content[:300]}")

        # Check similarity
        if _is_too_similar_to_recent(content, log_path, threshold=0.45):
            print("  [!] Too similar to recent — would retry in production")

        # Actually post to Moltbook
        try:
            result = send_post_to_moltbook(title=title, content=content)
            if "error" in str(result).lower() and "rate_limit" in str(result).lower():
                print(f"  [RATE LIMITED] Cooldown active")
                rate_limited += 1
            else:
                posted += 1
                print(f"  [POSTED] ID: {result.get('id', 'unknown')}")
        except Exception as e:
            err = str(e)
            if "429" in err:
                print(f"  [RATE LIMITED] 429 — waiting...")
                rate_limited += 1
            else:
                print(f"  [ERROR] {err[:200]}")

        # Wait between posts (rate limit cooldown)
        if i < count - 1:
            wait = 35 if rate_limited == 0 else 60
            print(f"  Waiting {wait}s before next post...")
            time.sleep(wait)

    print(f"\n{SEPARATOR}")
    print(f"POST RESULTS: {posted} posted, {rate_limited} rate-limited out of {count}")
    print(SEPARATOR)
    return posted


def test_interactions(target_comments=10):
    """Upvote and comment on feed posts."""
    my_name = _env("MOLTBOOK_AGENT_NAME", "NullArchitect").lower()

    print(f"\n{SEPARATOR}")
    print(f"FETCHING FEED...")
    print(SEPARATOR)

    posts = fetch_feed(limit=20)
    if not posts:
        print("[!] No posts in feed")
        return

    others = [p for p in posts if (p.get("author", {}).get("name") or "").lower() != my_name]
    print(f"Found {len(others)} posts from other agents")

    already_commented = _load_commented_posts()
    commented = 0
    upvoted = 0

    for post in others:
        pid = post["id"]
        title = post.get("title", "?")
        content = post.get("content", "")
        author = post.get("author", {}).get("name", "?")

        print(f"\n--- Post by {author}: \"{title[:50]}\"")

        # Upvote everything
        print(f"  Upvoting...")
        result = upvote_post(pid)
        if "error" not in result:
            upvoted += 1
            print(f"  [UPVOTED]")
        else:
            print(f"  [upvote skip] {result.get('error', '')}")
        time.sleep(2)

        # Comment if not already commented and we need more
        if commented < target_comments and pid not in already_commented and len(content) > 50:
            comment_text = _generate_comment(title, content)
            print(f"  Commenting: \"{comment_text[:120]}\"")
            result = comment_on_post(pid, comment_text)
            if "error" not in result:
                commented += 1
                _save_commented_post(pid)
                print(f"  [COMMENTED] ({commented}/{target_comments})")
            else:
                print(f"  [comment error] {result.get('error', '')}")
            time.sleep(5)

        # Follow
        if author != "?" and author.lower() != my_name:
            follow_agent(author)

        if commented >= target_comments:
            print(f"\n  Reached {target_comments} comments target.")
            break

    print(f"\n{SEPARATOR}")
    print(f"INTERACTION RESULTS: {upvoted} upvotes, {commented} comments")
    print(SEPARATOR)


if __name__ == "__main__":
    print("=" * 60)
    print("FULL INTERACTION TEST")
    print("10 posts + 10 comments + upvotes")
    print("=" * 60)

    # Phase 1: Post
    print("\n>>> PHASE 1: POSTING")
    posted = test_posts(count=10)

    # Phase 2: Interact
    print("\n>>> PHASE 2: INTERACTIONS (upvote + comment)")
    test_interactions(target_comments=10)

    print("\n>>> TEST COMPLETE")
