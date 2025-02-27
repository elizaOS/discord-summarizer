# 💻-coders 2024-11-12

## Summary
The main technical discussions revolved around the imageDescriptionService and its compatibility with Twitter client, as well as potential discord bot permission adjustments. A solution for an error after editing env was sought by @nftranch.

## FAQ
- Did anyone manage to make the telegram client work? (00:02)? (asked by @SotoAlt | WAWE)
- Has anyone a working version of imageDescriptionService and imageGenerationPlugin?(1.10) (asked by Dorian)
- Which API should I use? - @walee recommended OpenAI AI for me, but it only allows one message to be sent and then waits before sending another response. Is there a way around this limitation? (asked by only1)
- What frequency did you set your tweet automation at initially? - @ashx mentioned that the default was on defaults, but they changed it to 60-240 minutes. (asked by H.D.P.)
- How to handle multiple Twitter accounts in one instance? Can anyone guide me through it and how can we generate posts directly on Twitter using this setup? (asked by only1 (02:39))
- What voice model should I set for the eleven labs voice clone API key in .env file? (https://discord.com/channels/1253563208833433701/1301608253410775132/1305466629177606195) (asked by @Mahditwentytwo)
- Having trouble getting Bot to appear in Discord private server. Any luck with that? (asked by @björn)
- How do I chat with the agent? pnpm run shell is saying can't find shell. (asked by [Ophiuchus](05:17))
- Why am I unable to use Anthropic model and what causes the vercel.ai error? (asked by [SMA](05:20, SotoAlt | WAWE)(DM open))
- Why aren't messages appearing in terminal when running the bot? What should I do? (asked by [SotoAlt | WAWE](06:39))

## Who Helped Who
- @SotoAlt helped nftranch with Finding a solution for the error after editing env by providing @ferric | stakeware.xyz was mentioned by @weizguy (00:10)
- @ferric helped @Dorian with Looking at another version of code to solve rate limit issues by providing @vivoidos | bubbacat was mentioned by @Oguz Serdar (00:24)
- walee helped only1 with Recommended an API for chatbot integration and provided a code snippet with adjusted tweet frequency settings by providing @H.D.P.
- @Adnan helped @SMA with Understanding of twitter interactions in codebase. by providing Adnan helped SMA understand how the Twitter part works, specifically handleTwitterInteractionsLoop() function.
- @0xfabs helped @only1 with Setting up AI integration in IDE. by providing '0xfabs' offered to help 'only1' set up Cursor as a drop-in replacement for VSCode.
- @hiroP helped @only1 with Clarification on original messages handling in Twitter interactions. by providing 'hiroP' clarified the misunderstanding about post.ts and provided a reference to ash's js code for easy configuration.
- [SMA] helped [Stacker✝] (DM open) with AI Agent Deployment by providing [bundo](05:41, bunto) offered help with deploying an AI agent
- [Slim](06:35) helped [SotoAlt | WAWE] with Telegram Bot Setup by providing Slid provided guidance on setting up Telegram bot token, adding clients in defaultCharacter.ts
- @WAWE (07:38) helped @andy8052 with Using an alternative language model to simplify the process of creating a template with characters. by providing SotoAlt | WAWE provided guidance on using another LLM for character templates
- [degendata] helped [JARS] with Eliza Setup by providing [DeGenData](08:26) provides a YouTube tutorial for Eliza setup in response to jars' query.

## Action Items

### Technical Tasks
- Investigate why imageDescriptionService is not working with Twitter client (mentioned by @Dorian)
- Implement a tweet frequency adjustment (mentioned by @ashx)
- Setup Cursor as a drop-in replacement for VSCode to integrate AI directly into IDE (mentioned by @0xfabs)
- Configure post.ts settings in the js codebase, as per ash's modifications for easy configuration of original messages (mentioned by @hiroP)
- Assist SMA with setting voice model in character card API key via .env file, using eleven labs settings as reference (mentioned by @Mahditwentytwo)
- Provide technical help/guidance to thevelopers.eu team for building LLM matching algorithms and AI Agent OS (mentioned by @iwo | theVelopers.eu ✔️)
- Troubleshoot Bot appearance in private Discord servers, as reported by björn (mentioned by @björn)
- Resolve issue with using Anthropic's model via API within characterdefault.ts file (mentioned by [SMA](05:22))
- Deploy an AI agent for Bundo's project (if spare time available) (mentioned by [bundo](05:41, bunto)(DM open))
- Add import statements for TelegramClient (mentioned by [SotoAlt | WAWE](06:38))
- Install Cursor for Discord bot (mentioned by @jars)
- Install NodeJS, PNPM (mentioned by [jars](08:03))
- Lev needs help with API key setup for together.xyz (mentioned by @SotoAlt | WAWE)
- Micheal requires assistance in setting up Eliza on Docker. (mentioned by @o5l310)

### Documentation Needs
- Check if discord bot permissions need adjusting in the dashboard (asked by @hiroP) (mentioned by @only1)
- Review and potentially refactor the code for handling interactions in Discord chat. (mentioned by @H.D.P.)
- Share Telegram setup for API and client usage (mentioned by [SotoAlt | WAWE](05:35, SMA)(DM open))
- Watch Cursor for Newbies video tutorial on development environment setup. (mentioned by [SotoAlt | WAWE](08:04))

### Feature Requests
- Monitor AI agent for potential token deployment (mentioned by [Stacker✝](05:44-SotoAlt | WAWE)(DM open))
- Give bot admin access in group chat to receive messages from Telegram Bot (mentioned by [Slim](06:41))